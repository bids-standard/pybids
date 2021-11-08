#!/bin/bash

# small bash script to create a synthethic BIDS data set

# defines where the BIDS data set will be created
start_dir=$(pwd)
raw_dir=${start_dir}/../

subject_list='01 02 03 04 05'
session_list='01'

create_raw_dwi() {

	target_dir=$1
	subject=$2
	ses=$3

	this_dir=${target_dir}/sub-${subject}/ses-${ses}/dwi

	mkdir -p ${this_dir}

	suffix='_dwi'
	source_image=${target_dir}/../images/4d.nii.gz
	filename=${this_dir}/sub-${subject}_ses-${ses}${suffix}.nii.gz
	cp ${source_image} ${filename}

	source_bval=${start_dir}/dwi.bval
	filename=${this_dir}/sub-${subject}_ses-${ses}${suffix}.bval
	cp ${source_bval} ${filename}

	source_bvec=${start_dir}/dwi.bvec
	filename=${this_dir}/sub-${subject}_ses-${ses}${suffix}.bvec
	cp ${source_bvec} ${filename}

}

# RAW DATASET
for subject in ${subject_list}; do
	for ses in ${session_list}; do
		create_raw_dwi ${raw_dir} ${subject} ${ses}
	done

done
