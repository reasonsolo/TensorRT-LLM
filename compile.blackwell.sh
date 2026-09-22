#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -uex

PARTITION=${1:-batch}
CLEAN=${2:---clean}

ACCOUNT=${ACCOUNT:-coreai_comparch_inferencex}
REPO_DIR=$(readlink -f "$PWD")

# Mount layout is cluster-specific. hecate keeps everything under /lustre and
# the build has historically mounted the whole filesystem in. oci-jhb has no
# /lustre/share at all -- there /lustre is a symlink to /scratch and the real
# lustre mount is /scratch/fsw -- so "/lustre:/lustre" resolves to a target the
# base image does not have. enroot then auto-creates the missing target as a
# directory the submitting user cannot traverse, and that single directory makes
# --container-save abort with "find: Permission denied / pyxis: failed to export
# container", leaving a successful build with no image (measured on hecate
# 2026-09-18, see compile.sh). /workspace exists in the base image, and the build
# itself needs nothing but the repo: every path in the container command below is
# relative to WORKDIR.
#
# The /lustre/share probe is the same marker scripts/run_branch_guard_perf_k3.py
# uses to tell the two clusters apart; see _detect_cluster() there.
if [[ -d /lustre/share ]]; then
    MOUNT="/lustre:/lustre,${REPO_DIR}:/code/tensorrt_llm"
    WORKDIR="/code/tensorrt_llm"
else
    MOUNT="${REPO_DIR}:/workspace"
    WORKDIR="/workspace"
fi
MOUNT=${TRTLLM_BUILD_MOUNT:-${MOUNT}}
WORKDIR=${TRTLLM_BUILD_WORKDIR:-${WORKDIR}}

source "${REPO_DIR}/jenkins/current_image_tags.properties"
IMAGE=${TRTLLM_TEST_IMAGE:-${LLM_SBSA_DOCKER_IMAGE}}

HASH=$(git -C "${REPO_DIR}" rev-parse --short HEAD)
BUILD_DIR="${REPO_DIR}/build_images/${HASH}-blackwell"
JOB_NAME="${ACCOUNT}-blackwell-build.${HASH}"

if [[ -d ${BUILD_DIR} ]]; then
    rm -r "${BUILD_DIR}"
fi
mkdir -p "${BUILD_DIR}"

rm "${REPO_DIR}"/build/tensorrt_llm-*.whl || true

# PIN OVERRIDE REMOVED -- do not re-add from the rubin-advance version of this
# script. rubin-advance force-reinstalls flashinfer-python==0.6.16 and
# nvidia-cutlass-dsl[cu13]==4.7.1 between the requirements-dev.txt install and
# build_wheel.py. Those versions were chosen against rubin-advance's base
# image; this branch resolves a different base image from its own
# jenkins/current_image_tags.properties and pins its own versions in
# requirements.txt. Let requirements.txt decide -- nothing is force-reinstalled.
#
# --upgrade REMOVED from the requirements-dev.txt install. On this branch's base
# image "pip install --upgrade -r requirements-dev.txt" fails outright:
# requirements-dev.txt asks for pyyaml>=6.0.1,<6.0.3, the image ships an
# apt-managed PyYAML 6.0.1 that already satisfies it, and only --upgrade makes
# pip try to move to 6.0.2 -- which needs to uninstall a debian-installed
# package with no RECORD file ("error: uninstall-no-record-file"). The in-repo
# scripts (jenkins/Build.groovy, jenkins/L0_Test.groovy,
# jenkins/scripts/slurm_install.sh) all install requirements-dev.txt without
# --upgrade; they are the authority on build flags.
#
# Deliberately NOT using --ignore-installed: that flag is global to the pip
# invocation, not scoped to one package, so it would replace the container's
# CUDA-enabled torch with a CPU-only PyPI wheel.

srun -A "${ACCOUNT}" -p "${PARTITION}" --job-name="${JOB_NAME}" --container-image="${IMAGE}" -G 4 \
    --container-mounts="${MOUNT}" \
    --container-workdir="${WORKDIR}" \
    --container-save="${BUILD_DIR}/trtllm.sqsh" \
    -N 1 --ntasks=1 --ntasks-per-node=1 -t 2:00:00 \
    bash -c "python3 -m pip install -r requirements-dev.txt && python scripts/build_wheel.py -G Ninja -a '103-real' ${CLEAN} && pip install build/tensorrt_llm-*.whl"

cp "${REPO_DIR}"/build/tensorrt_llm-*.whl "${BUILD_DIR}/"
