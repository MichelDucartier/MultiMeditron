#!/bin/bash

set -euo pipefail

cd $(dirname $0)

set -x

podman build -t $USER/ngc-multimeditron:24.10 -f Dockerfile ..
enroot import -x mount -o $CAPSCRATCH/images/ngc-multimeditron+24.10.sqsh podman://$USER/ngc-multimeditron:24.10

set +x
