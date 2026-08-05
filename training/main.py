# Main executable file

import argparse
from email import parser
import torch
import torch.optim as optim
from yaml import parser
from torch.utils.data import DataLoader, Subset

from models import MultiTaskLoss, DualBranchEfficientNet, BaselineVariantCNN, BaselineAirlineCNN, DualBranchResNet, DualBranchConvNeXt, DualBranchViT, SingleTaskNet
from dataset import AirlinerDataset, load_split_dataframes, build_mapping_from_csv, get_transforms
from train import train_net, evaluate, test_evaluate
from utils import load_model_checkpoint, plot_training_curve, predict_image, get_model_name

def main():
    parser = argparse.ArgumentParser(description="Airliner Image Classification")
    
    # Mode and Setup
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'sanity', 'predict', 'test', 'graph'], help="Execution mode")    
    parser.add_argument('--model', type=str, required=True, 
                        choices=['baseline_variant', 'baseline_airline', 'primary_efficientnet', 'primary_resnet', 
                                 'primary_convnext', 'primary_vit', 'single_variant_convnext', 'single_airline_convnext', 
                                 'single_variant_efficientnet', 'single_airline_efficientnet'], 
                        help="Model architecture")    
    parser.add_argument('--is_multitask', action='store_true', help="Use if the model outputs multiple branches (e.g. Primary)")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default=r"../Data/airliners_images")
    parser.add_argument('--train_csv', type=str, default=r"../Data/metadata/train/train_metadata.csv")
    parser.add_argument('--val_csv', type=str, default=r"../Data/metadata/val/val_metadata.csv")
    parser.add_argument('--test_csv', type=str, default=r"../Data/metadata/test/test_metadata.csv")
    parser.add_argument('--airline_csv', type=str, default=r"../Data/metadata/counts_airlines_merged_trimmed.csv")
    parser.add_argument('--variant_csv', type=str, default=r"../Data/metadata/counts_variants_trimmed.csv")

    parser.add_argument('--num_workers', type=int, default=4, help="Number of CPU workers for the DataLoader")
    
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--image_size', type=int, default=300) # 224 for ablation study of ViT, back to 300 for regular CNNs
    
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay for AdamW")
    parser.add_argument('--hidden_dim', type=int, default=512, help="Hidden dimension for MLP heads")
    parser.add_argument('--dropout_rate', type=float, default=0.3, help="Dropout rate for MLP heads")
    parser.add_argument('--use_scheduler', action='store_true', help="Enable ReduceLROnPlateau scheduler")
    
    # Logging and checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints")
    parser.add_argument('--resume_checkpoint', type=str, default=None, help="Path to checkpoint to resume/predict from")
    parser.add_argument('--checkpoint_freq', type=int, default=1)
    parser.add_argument('--record_freq', type=int, default=100)
    parser.add_argument('--track_iters', action='store_true', help="Enable iteration-level tracking for granular graphs")
    parser.add_argument('--run_name', type=str, default=None, help="Custom name for this run/model (overrides default model name)")
    
    # Inference specific
    parser.add_argument('--image_path', type=str, default=None, help="Image to predict (for predict mode)")
    parser.add_argument('--output_test_csv', action='store_true', help="Output per-class accuracy CSVs during test mode")
    parser.add_argument('--test_csv_dir', type=str, default="test_results", help="Directory to save test CSVs")

    # Plotting

    args = parser.parse_args()

    # Primary models are multitask by design
    if 'primary' in args.model:
        args.is_multitask = True


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Build Metadata Mappings
    airline_mapping = build_mapping_from_csv(args.airline_csv, column_name='Airline')
    variant_mapping = build_mapping_from_csv(args.variant_csv, column_name='Variant')
    NUM_VARIANT_CLASSES = len(variant_mapping)
    NUM_AIRLINE_CLASSES = len(airline_mapping)

    # 2. Instantiate Model
    if args.model == 'baseline_variant':
        model = BaselineVariantCNN(num_variant_classes=NUM_VARIANT_CLASSES).to(device)
    elif args.model == 'baseline_airline':  
        model = BaselineAirlineCNN(num_airline_classes=NUM_AIRLINE_CLASSES).to(device)
    elif args.model == 'primary_efficientnet':
        model = DualBranchEfficientNet(num_variant_classes=NUM_VARIANT_CLASSES, num_airline_classes=NUM_AIRLINE_CLASSES).to(device)
    elif args.model == 'primary_resnet':
        model = DualBranchResNet(num_variant_classes=NUM_VARIANT_CLASSES, num_airline_classes=NUM_AIRLINE_CLASSES).to(device)
    elif args.model == 'primary_convnext':
        model = DualBranchConvNeXt(num_variant_classes=NUM_VARIANT_CLASSES, num_airline_classes=NUM_AIRLINE_CLASSES,
                                   hidden_dim=args.hidden_dim, dropout_rate=args.dropout_rate).to(device)
    elif args.model == 'primary_vit':
        model = DualBranchViT(num_variant_classes=NUM_VARIANT_CLASSES, num_airline_classes=NUM_AIRLINE_CLASSES).to(device)
    elif args.model == 'single_variant_efficientnet':
        model = SingleTaskNet(backbone_type="efficientnet", num_classes=NUM_VARIANT_CLASSES, task_name="Variant").to(device)
    elif args.model == 'single_airline_efficientnet':
        model = SingleTaskNet(backbone_type="efficientnet", num_classes=NUM_AIRLINE_CLASSES, task_name="Airline").to(device)
    elif args.model == 'single_variant_convnext':
        model = SingleTaskNet(backbone_type="convnext", num_classes=NUM_VARIANT_CLASSES, task_name="Variant").to(device)
    elif args.model == 'single_airline_convnext':
        model = SingleTaskNet(backbone_type="convnext", num_classes=NUM_AIRLINE_CLASSES, task_name="Airline").to(device)

    
    # 2.5 Instantiate MultiTaskLoss and AdamW Optimizer
    multi_task_loss = None
    if args.is_multitask:
        multi_task_loss = MultiTaskLoss(num_tasks=2).to(device)
        # Pass BOTH model parameters and loss parameters to AdamW
        optimizer = optim.AdamW(list(model.parameters()) + list(multi_task_loss.parameters()), lr=args.lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Determine the target task for single-task models
    target_task = "variant" # default
    if not args.is_multitask:
        if "airline" in args.model:
            target_task = "airline"
    
    start_epoch, loaded_history = 0, None
    if args.resume_checkpoint:
        model, optimizer, start_epoch, loaded_history = load_model_checkpoint(
            args.resume_checkpoint, model, optimizer, device, multi_task_loss=multi_task_loss
        )

    # 3. Execution logic branching
    if args.mode in ['train', 'sanity']:
        train_df, val_df, test_df = load_split_dataframes(args.train_csv, args.val_csv, args.test_csv)
        train_transforms, eval_transforms = get_transforms(args.image_size)

        train_dataset = AirlinerDataset(train_df, args.data_dir, airline_mapping, variant_mapping, transform=train_transforms)
        val_dataset = AirlinerDataset(val_df, args.data_dir, airline_mapping, variant_mapping, transform=eval_transforms)
        
        if args.mode == 'sanity':
            # Run on small subset of data
            train_dataset = Subset(train_dataset, range(16))
            val_dataset = train_dataset
            args.batch_size = min(4, args.batch_size)
            args.record_freq = 2

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        saved_multi_path = train_net(
            net=model,
            train_loader=train_loader,
            val_loader=val_loader,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            is_multitask=args.is_multitask,
            checkpoint_freq=args.checkpoint_freq,
            checkpoint_dir=args.checkpoint_dir,
            start_epoch=start_epoch,
            loaded_history=loaded_history,
            track_iterations=args.track_iters,
            record_freq=args.record_freq,
            custom_model_name=args.run_name,
            num_workers=args.num_workers,
            multi_task_loss=multi_task_loss,
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate,
            weight_decay=args.weight_decay, 
            use_scheduler=args.use_scheduler,     
            target_task=target_task
        )
        
        # In the 'train' block, change the plot_training_curve call to include the save_path_prefix:
        plot_training_curve(saved_multi_path, is_multitask=args.is_multitask, target_task=target_task, save_path_prefix=saved_multi_path)
        
    elif args.mode == 'predict':
        if not args.image_path:
            raise ValueError("Must provide --image_path in predict mode.")
            
        variant_idx_to_name = {v: k for k, v in variant_mapping.items()}
        airline_idx_to_name = {v: k for k, v in airline_mapping.items()}

        # Predict image now returns lists of tuples: [(class_name, probability), ...]
        var_results, air_results = predict_image(
            image_path=args.image_path,
            model=model,
            device=device,
            variant_idx_to_name=variant_idx_to_name,
            airline_idx_to_name=airline_idx_to_name,
            is_multitask=args.is_multitask,
            image_size=args.image_size,
            top_k=5,
            show_plot=True # Set this to False if you don't want the pop-up window
        )
        
        print("\n" + "="*30)
        print("       TOP 5 PREDICTIONS")
        print("="*30)
        
        if args.is_multitask:
            print("\n--- VARIANT ---")
            for i, (name, prob) in enumerate(var_results):
                print(f"  {i+1}. {name:<25} {prob*100:>6.2f}%")
                
            print("\n--- AIRLINE ---")
            for i, (name, prob) in enumerate(air_results):
                print(f"  {i+1}. {name:<25} {prob*100:>6.2f}%")
        else:
            print("\n--- PREDICTIONS ---")
            for i, (name, prob) in enumerate(var_results):
                print(f"  {i+1}. {name:<25} {prob*100:>6.2f}%")
        print("="*30 + "\n")

    elif args.mode == 'test':
        # Load just the test dataset
        _, _, test_df = load_split_dataframes(args.train_csv, args.val_csv, args.test_csv)
        _, eval_transforms = get_transforms(args.image_size)
        test_dataset = AirlinerDataset(test_df, args.data_dir, airline_mapping, variant_mapping, transform=eval_transforms)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        # We must have a trained model to test!
        if not args.resume_checkpoint:
            raise ValueError("Must provide --resume_checkpoint in test mode.")
            
        criterion = torch.nn.CrossEntropyLoss()
        
        # Reverse the mappings for CSV output
        variant_idx_to_name = {v: k for k, v in variant_mapping.items()}
        airline_idx_to_name = {v: k for k, v in airline_mapping.items()}
        
        run_name = args.run_name if args.run_name else args.model

        if args.is_multitask:
            test_var_f1, test_air_f1, test_loss = test_evaluate(
                net=model, loader=test_loader, criterion=criterion, device=device,
                variant_idx_to_name=variant_idx_to_name, airline_idx_to_name=airline_idx_to_name,
                is_multitask=True, target_task=target_task, 
                output_csv=args.output_test_csv, csv_dir=args.test_csv_dir, run_name=run_name
            )
            print(f"Final Test - Var F1: {test_var_f1:.4f} | Air F1: {test_air_f1:.4f} | Loss: {test_loss:.4f}")
        else:
            test_f1, test_loss = test_evaluate(
                net=model, loader=test_loader, criterion=criterion, device=device,
                variant_idx_to_name=variant_idx_to_name, airline_idx_to_name=airline_idx_to_name,
                is_multitask=False, target_task=target_task, 
                output_csv=args.output_test_csv, csv_dir=args.test_csv_dir, run_name=run_name
            )
            print(f"Final Test - {target_task.capitalize()} F1: {test_f1:.4f} | Loss: {test_loss:.4f}")

    elif args.mode == 'graph':
        if not args.resume_checkpoint:
            raise ValueError("Must provide --resume_checkpoint to extract model metadata for graphing.")
        
        import os
        # Load the checkpoint to read metadata
        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        metadata = checkpoint.get("metadata", {})
        
        if metadata:
            model_name = metadata.get("model_name", args.model)
            bs = metadata["hyperparameters"]["batch_size"]
            lr = metadata["hyperparameters"]["learning_rate"]
            is_multitask = metadata.get("is_multitask", args.is_multitask)
            
            # Extract new parameters with safe defaults for older checkpoints
            wd = metadata["hyperparameters"].get("weight_decay", 0.01)
            hd = metadata["hyperparameters"].get("hidden_dim", 512)
            drop = metadata["hyperparameters"].get("dropout_rate", 0.3)
            use_sch = metadata["hyperparameters"].get("use_scheduler", False)
            
        else:
            print("Warning: No metadata found in checkpoint. Falling back to argparse flags.")
            model_name = args.run_name if args.run_name else args.model
            bs = args.batch_size
            lr = args.lr
            is_multitask = args.is_multitask
            
            # Fall back to the argparse flags if metadata is missing
            wd = args.weight_decay
            hd = args.hidden_dim
            drop = args.dropout_rate
            use_sch = args.use_scheduler
            
        epoch = checkpoint.get("epoch", "unknown")

        # 1. Reconstruct the base path for where the CSVs were saved (always saved with 'final')
        csv_base_path = get_model_name(model_name, bs, lr, wd, hd, drop, use_sch, "final", args.checkpoint_dir).replace('.pt', '')    

        # 2. Reconstruct the file name prefix for the PNGs
        save_prefix = os.path.join(args.checkpoint_dir, f"graph_{model_name}_epoch{epoch}")
        
        print(f"Generating graphs using CSVs located at: {csv_base_path}*.csv (Up to epoch {args.epochs})")
        
        # 3. Determine single-task target
        target_task = "variant"
        if not is_multitask and "airline" in model_name.lower():
            target_task = "airline"
            
        # Pass args.epochs as max_epoch
        plot_training_curve(csv_base_path, is_multitask=is_multitask, target_task=target_task, save_path_prefix=save_prefix, max_epoch=args.epochs)

if __name__ == "__main__":
    main()


# IDEAS: output metrics in csvs for each class in test mode
# for predict mode, output loss and/or top 5 predicated classes with probabilities, and maybe a visualization of the image with the predicted class highlighted.