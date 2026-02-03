""" Script to download checkpoint and data for FollowMyHold """

set -euo pipefail

# HaMeR ckpts from https://github.com/geopavlakos/hamer/blob/main/fetch_demo_data.sh
gdown https://drive.google.com/uc?id=1mv7CUAnm73oKsEEG1xE3xH2C_oqcFSzT
tar --warning=no-unknown-keyword --exclude=".*" -xvf hamer_demo_data.tar.gz
mv _DATA third_party/estimator/hamer

# download WiLoR hand detector model from https://github.com/rolpotamias/WiLoR
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P ./third_party/estimator/wilor_ckpt/
export WILOR_CKPT="./third_party/estimator/wilor_ckpt/detector.pt"
