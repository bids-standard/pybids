#!/bin/bash

# small bash script to create a synthetic BIDS data set

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

	mkdir -p "${this_dir}"

	suffix='_dwi'
	source_image=${target_dir}/../images/4d.nii.gz
	filename=${this_dir}/sub-${subject}_ses-${ses}${suffix}.nii.gz
	cp "${source_image}" "${filename}"

	source_bval=${start_dir}/dwi.bval
	filename=${this_dir}/sub-${subject}_ses-${ses}${suffix}.bval
	cp "${source_bval}" "${filename}"

	source_bvec=${start_dir}/dwi.bvec
	filename=${this_dir}/sub-${subject}_ses-${ses}${suffix}.bvec
	cp "${source_bvec}" "${filename}"

}

create_raw_fmap() {

	target_dir=$1
	subject=$2
	ses=$3

	this_dir=${target_dir}/sub-${subject}/ses-${ses}/fmap

	mkdir -p ${this_dir}

	source_image=${target_dir}/../images/3d.nii.gz

	fmap_suffix_list='_phasediff _magnitude1 _magnitude2'

	for suffix in ${fmap_suffix_list}; do
		cp "${source_image}" "${this_dir}/sub-${subject}_ses-${ses}${suffix}.nii.gz"
	done

	RepetitionTime=0.4
	EchoTime1=00519
	EchoTime2=0.00765
	FlipAngle=60
	template='{"FlipAngle":%f, "RepetitionTime":%f, "EchoTime1":%f, "EchoTime2":%f, "IntendedFor": ["%s", "%s"], "PhaseEncodingDirection": "j-"}'

	suffix='_bold'

	task_name='nback'
	IntendedFor1="$(echo ses-${ses}/func/sub-${subject}_ses-${ses}_task-${task_name}_run-01${suffix}.nii.gz)"
	IntendedFor2="$(echo ses-${ses}/func/sub-${subject}_ses-${ses}_task-${task_name}_run-02${suffix}.nii.gz)"
	json_string=$(printf "$template" "$FlipAngle" "$RepetitionTime" "$EchoTime1" "$EchoTime2" "$IntendedFor1" "$IntendedFor2")
	echo "$json_string" >${this_dir}/sub-${subject}_ses-${ses}_phasediff.json
	echo "$json_string" >${this_dir}/sub-${subject}_ses-${ses}_phasediff.json

}

# RAW DATASET
for subject in ${subject_list}; do
	for ses in ${session_list}; do
		create_raw_dwi ${raw_dir} ${subject} ${ses}
		create_raw_fmap ${raw_dir} ${subject} ${ses}
	done

done
