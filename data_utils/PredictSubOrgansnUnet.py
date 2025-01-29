"""
Use this to run the nnU-Net inference. You can run multiple instances of this code in parallel!
Just change the `part_id` and `gpu` arguments to split the task into multiple parts.
Example of splitting it in 4 parts:
    python PredictSubOrgansnUnet.py --num_parts 4 --part_id 0 --gpu 0
    python PredictSubOrgansnUnet.py --num_parts 4 --part_id 1 --gpu 1
    python PredictSubOrgansnUnet.py --num_parts 4 --part_id 2 --gpu 2
    python PredictSubOrgansnUnet.py --num_parts 4 --part_id 3 --gpu 3
"""


import argparse
from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import os

def split_files(files_input, files_output, num_parts, part_id):
    """
    Splits the files_input and files_output into num_parts and selects the partition for the given part_id.

    Args:
        files_input (list): List of input files.
        files_output (list): List of output files.
        num_parts (int): Number of partitions.
        part_id (int): Partition ID to select (starting from 0).

    Returns:
        tuple: (files_input_partition, files_output_partition) for the given part_id.
    """
    assert len(files_input) == len(files_output), "files_input and files_output must have the same length"
    assert 0 <= part_id < num_parts, "Invalid part_id. It must be in the range [0, num_parts-1]"

    # Split the files based on num_parts
    total_files = len(files_input)
    files_per_part = (total_files + num_parts - 1) // num_parts  # Ensure rounding up
    start_idx = part_id * files_per_part
    end_idx = min(start_idx + files_per_part, total_files)

    # Return the partition
    return files_input[start_idx:end_idx], files_output[start_idx:end_idx]

def parse_args():
    parser = argparse.ArgumentParser(description="Run nnUNet prediction with custom arguments.")
    parser.add_argument('--pth', type=str, default='/mnt/ccvl15/psalvad2/nnUNet_raw/Dataset244_smallAtlasUCSF/imagesTr/', 
                        help="Path to input images folder")
    parser.add_argument('--outdir', type=str, default='/mnt/ccvl15/psalvad2/Dataset244_smallAtlasUCSF_suborgans/', 
                        help="Path to output folder for predictions")
    parser.add_argument('--checkpoint', type=str, 
                        default='/mnt/ccvl15/psalvad2/nnUNetOrgansAndSubSegments', 
                        help="Path to model checkpoint folder")
    parser.add_argument('--num_parts', type=int, default=1, 
                        help="Number of parts to split the task into")
    parser.add_argument('--part_id', type=int, default=0, 
                        help="ID of the current part (0-indexed)")
    parser.add_argument('--gpu', type=int, default=0, 
                        help="ID of the gpu to be used")
    parser.add_argument('--workers', type=int, default=2, 
                        help="Number of worker processes for preprocessing and segmentation export")
    return parser.parse_args()


def main():
    args = parse_args()

    # Instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', args.gpu),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    # Collect input and output files
    files_input = [[os.path.join(args.pth, file)] for file in sorted(os.listdir(args.pth)) if file.endswith('.nii.gz')]
    files_output = [os.path.join(args.outdir, file.replace('.nii.gz','')) for file in sorted(os.listdir(args.pth)) if file.endswith('.nii.gz')]
    files_input, files_output = split_files(files_input, files_output, args.num_parts, args.part_id)



    # Initialize the network architecture and load the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, args.checkpoint),
        use_folds=('all',),
        checkpoint_name='checkpoint_final.pth',
    )

    # Run the prediction
    predictor.predict_from_files(
        files_input, files_output,
        save_probabilities=False,
        overwrite=False,
        num_processes_preprocessing=args.workers,
        num_processes_segmentation_export=args.workers,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )


if __name__ == '__main__':
    main()