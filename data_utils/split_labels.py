import os
import glob
import argparse
import numpy as np
import nibabel as nib
from concurrent.futures import ProcessPoolExecutor

labels={'background': 0,
            'kidney_right': 1,
            'kidney_left': 2,
            'kidney_lesion': 3,
            'kidney_lesion_kidney_right': 4,
            'kidney_lesion_kidney_left': 5,
            'pancreas': 6,
            'pancreas_head': 7,
            'pancreas_body': 8,
            'pancreas_tail': 9,
            'pancreatic_lesion': 10,
            'pancreatic_lesion_pancreas_head': 11,
            'pancreatic_lesion_pancreas_body': 12,
            'pancreatic_lesion_pancreas_tail': 13,
            'liver': 14,
            'liver_segment_1': 15,
            'liver_segment_2': 16,
            'liver_segment_3': 17,
            'liver_segment_4': 18,
            'liver_segment_5': 19,
            'liver_segment_6': 20,
            'liver_segment_7': 21,
            'liver_segment_8': 22,
            'liver_lesion': 23,
            'liver_lesion_liver_segment_1': 24,
            'liver_lesion_liver_segment_2': 25,
            'liver_lesion_liver_segment_3': 26,
            'liver_lesion_liver_segment_4': 27,
            'liver_lesion_liver_segment_5': 28,
            'liver_lesion_liver_segment_6': 29,
            'liver_lesion_liver_segment_7': 30,
            'liver_lesion_liver_segment_8': 31,}

# Classes we want to output explicitly
out_labels = [
    "liver",
    "pancreas",
    "kidney_right",
    "kidney_left",
    "liver_lesion",
    "kidney_lesion",
    "pancreatic_lesion",
    "pancreas_head",
    "pancreas_body",
    "pancreas_tail"
]
# Add "liver_segment_1" through "liver_segment_8"
for i in range(1, 9):
    out_labels.append("liver_segment_" + str(i))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Split segmentation labels into multiple .nii.gz files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input .nii.gz files to process.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the splitted .nii.gz files will be saved.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel processes to use for splitting.",
    )
    return parser.parse_args()


def build_mapping():
    """
    Build the 'mapping' dictionary:
    Each key is an output class name,
    each value is a list of label IDs that should be merged into that class.
    """

    mapping = {}
    for key in out_labels:
        ids = []
        # Find all label IDs whose name contains `key`
        for name, val in labels.items():
            if key in name:
                ids.append(val)

        # Special case: "pancreas" also includes "pancreatic"
        if key == "pancreas":
            for name, val in labels.items():
                if "pancreatic" in name:
                    ids.append(val)

        mapping[key] = sorted(set(ids))

    return mapping



def split_segmentation(input_nifti_path, output_folder, classes_dict):
    """
    Reads a NIfTI segmentation, then saves separate .nii.gz masks
    for each class (merging labels if necessary).
    """
    seg_img = nib.load(input_nifti_path)
    # *** Fix: load as float, then cast to int16 (or directly from dataobj) ***
    seg_data = seg_img.get_fdata().astype(np.int16)
    # Or use: seg_data = np.asanyarray(seg_img.dataobj).astype(np.int16)

    # Ensure output_folder exists
    os.makedirs(output_folder, exist_ok=True)

    # We'll name output files using:
    #   <base of input file>_<class_name>.nii.gz
    base_name = os.path.basename(input_nifti_path)
    if base_name.endswith(".nii.gz"):
        base_name = base_name[:-7]
    else:
        base_name = os.path.splitext(base_name)[0]

    # Optionally create a subfolder for each input file
    this_out_folder = os.path.join(output_folder, base_name)
    os.makedirs(this_out_folder, exist_ok=True)

    for class_name, label_list in classes_dict.items():
        # Create binary mask: 1 if label in label_list, else 0
        mask = np.isin(seg_data, label_list).astype(np.uint8)

        mask_nifti = nib.Nifti1Image(mask, seg_img.affine, seg_img.header)
        out_file = os.path.join(this_out_folder, f"{class_name}.nii.gz")
        nib.save(mask_nifti, out_file)

    print(f"Finished splitting {input_nifti_path}!")

def process_file(input_path, output_dir, classes_dict):
    """Wrapper to process a single file."""
    split_segmentation(input_path, output_dir, classes_dict)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Build the label mapping once
    classes_dict = build_mapping()

    # Gather all .nii.gz files in the input directory
    nifti_files = glob.glob(os.path.join(args.input_dir, "*.nii.gz"))
    if len(nifti_files) == 0:
        print(f"No NIfTI files found in {args.input_dir}!")
        return

    # Use parallel processing
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for fpath in nifti_files:
            futures.append(
                executor.submit(process_file, fpath, args.output_dir, classes_dict)
            )
        for future in futures:
            future.result()

    print(f"All files processed. Output saved to: {args.output_dir}")

if __name__ == "__main__":
    main()