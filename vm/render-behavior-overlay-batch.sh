#!/bin/bash
#
#SBATCH --job-name=render-behavior
#
#SBATCH --qos=batch
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Example use of this SLURM script:
#   /projects/kumar-lab/USERS/sheppk/behavior-classification-env/render-behavior-overlay-batch.sh \
#       /fastscratch/sheppk/behavior-for-amelie-2020-05-04/batch.txt \
#       --video-root-dir /fastscratch/sheppk/behavior-for-amelie-2020-05-04 \
#       --behavior-root-dirs /fastscratch/sheppk/behavior-for-amelie-2020-05-04 \
#       --behavior Grooming \
#       --annotator-names Amelie \
#       --out-dir /fastscratch/sheppk/behavior-for-amelie-2020-05-04-out

module load python36

join_by_char() {
  local IFS="$1"
  shift
  echo "$*"
}

urldecode() {
    local decoded="$(python3 -c "import sys, urllib.parse as ul; print(ul.unquote_plus(sys.argv[1]))" "$1")"
    echo -n "$decoded"
}

urlencode() {
    local encoded="$(python3 -c "import sys, urllib.parse as ul; print (ul.quote_plus(sys.argv[1]))" "$1")"
    echo -n "$encoded"
}

serialize_arr() {
    local serialized=( )
    for (( i=1; i<=${#}; i++ ))
    do
        serialized[i]="$(urlencode "${@:$i:1}")"
    done

    echo -n "$(join_by_char "&" "${serialized[@]}")"
}

deserialize_arr() {
    deserialized=( )
    IFS='&' read -r -a deserialized <<< "$1"

    for (( i=0; i<${#deserialized[@]}; i++ ))
    do
        deserialized[i]="$(urldecode ${deserialized[$i]})"
    done
}

trim_sp() {
    local var="$*"
    # remove leading whitespace characters
    var="${var#"${var%%[![:space:]]*}"}"
    # remove trailing whitespace characters
    var="${var%"${var##*[![:space:]]}"}"
    echo -n "$var"
}

if [[ -z "${SLURM_JOB_ID}" ]]
then
    # the script is being run from command line. We should do a self-submit as an array job
    if [[ ( -f "${1}" ) ]]
    then
        # echo "${1} is set and not empty"
        echo "Preparing to submit rendering using batch file: ${1}"
        batch_line_count=$(wc -l < "${1}")
        echo "Submitting an array job for ${batch_line_count} videos"

        # Here we perform a self-submit
        echo sbatch --export=ROOT_DIR="$(dirname "${0}"),BATCH_FILE=${1},CMD_ARGS=$(serialize_arr "${@:2}")" --array="1-${batch_line_count}%500" "${0}"
        sbatch --export=ROOT_DIR="$(dirname "${0}"),BATCH_FILE=${1},CMD_ARGS=$(serialize_arr "${@:2}")" --array="1-${batch_line_count}%500" "${0}"
    else
        echo "ERROR: missing batch file." >&2
        echo "Expected usage:" >&2
        echo "render-behavior-overlay-batch.sh BATCH_FILE.txt OPTIONS_LIST" >&2
        exit 1
    fi
else
    # the script is being run by slurm
    if [[ -z "${SLURM_ARRAY_TASK_ID}" ]]
    then
        echo "ERROR: no SLURM_ARRAY_TASK_ID found" >&2
        exit 1
    fi

    if [[ -z "${BATCH_FILE}" ]]
    then
        echo "ERROR: the BATCH_FILE environment variable is not defined" >&2
        exit 1
    fi

    # echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
    # echo "BATCH_FILE: ${BATCH_FILE}"
    # echo "CMD_ARGS: ${CMD_ARGS}"

    deserialize_arr "${CMD_ARGS}"
    # echo "DESERIALIZED: ${deserialized[@]}"
    # for x in "${deserialized[@]}"
    # do
    #     echo "LINE ${x}"
    # done

    PROCESSING_ROOT_DIR="$(dirname "${BATCH_FILE}")"

    # here we use the array ID to pull out the right video from the batch file
    VIDEO_FILE=$(trim_sp $(sed -n "${SLURM_ARRAY_TASK_ID}{p;q;}" < "${BATCH_FILE}"))
    echo "BATCH VIDEO FILE: ${VIDEO_FILE}"

    # # the "v1" is for output format versioning. If format changes this should be updated
    # OUT_DIR="${VIDEO_FILE%.*}_behavior/v1"

    # cd "$(dirname "${BATCH_FILE}")"
    # POSE_FILE_V3="${VIDEO_FILE%.*}_pose_est_v3.h5"
    # POSE_FILE="${POSE_FILE_V3}"

    # if [[ ! ( -f "${POSE_FILE}" ) ]]
    # then
    #     POSE_FILE_V2="${VIDEO_FILE%.*}_pose_est_v2.h5"
    #     POSE_FILE="${POSE_FILE_V2}"
    # fi

    # if [[ ! ( -f "${POSE_FILE}" ) ]]
    # then
    #     echo "ERROR: failed to find either pose file (${POSE_FILE_V2} or ${POSE_FILE_V3}) for ${VIDEO_FILE}" >&2
    #     exit 1
    # fi

    echo "DUMP OF CURRENT ENVIRONMENT:"
    env
    echo "BEGIN PROCESSING: ${VIDEO_FILE}"
    module load singularity
    echo "${deserialized[@]}"
    echo singularity run "${ROOT_DIR}/render-behavior-overlay.sif" \
        --video-file "${VIDEO_FILE}" \
        "${deserialized[@]}"
    singularity run "${ROOT_DIR}/render-behavior-overlay.sif" \
        --video-file "${VIDEO_FILE}" \
        "${deserialized[@]}"

    echo "FINISHED PROCESSING: ${VIDEO_FILE}"
fi
