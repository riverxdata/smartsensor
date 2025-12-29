###########################################################################################################################
# simulate job directory
export BASEDIR=$PWD
export RIVER_HOME=$PWD/work
export job_id="job_id"
export PIXI_HOME=$RIVER_HOME/.pixi

mkdir -p $RIVER_HOME/jobs/job_id
cp $PWD/tests/params.json $RIVER_HOME/jobs/job_id/params.json
cd $RIVER_HOME/jobs/job_id

###########################################################################################################################
# === Install pixi ===
which pixi || curl -fsSL https://pixi.sh/install.sh | sh
export PATH=$PATH:$HOME/.pixi/bin
pixi config append default-channels bioconda --global
pixi config append default-channels conda-forge --global
pixi global install nextflow jq git singularity python=3.14

# Setup networking
export PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo $PORT > $RIVER_HOME/jobs/job_id/job.port
echo $(hostname) > $RIVER_HOME/jobs/job_id/job.host

# Load parameters and clone repository
while IFS== read -r key value; do
   export "$key=$value"
done < <(jq -r 'to_entries|map("\(.key)=\(.value|tostring)")|.[]' params.json)

# Create symlink to analysis directory
ln -sf $BASEDIR $RIVER_HOME/jobs/job_id/analysis

git=$(git remote get-url origin 2>/dev/null)
repo_name=$(basename -s .git "$git")
owner=$(basename "$(dirname "$git")")
local_dir="$RIVER_HOME/tools/$owner/$repo_name/$tag"

if [[ "$git" == *"nf-"* ]]; then
    profiles="${profile:+singularity,$profile}"
    profiles="${profiles:-singularity}"
    nextflow run "$owner/$repo_name" \
        -r "$tag" \
        -c river.config \
        -profile "$profiles" \
        -process.executor slurm \
        -process.shell 'bash' \
        --outdir "s3://$bucket_name/$outdir/job_id" \
        -with-report "s3://$bucket_name/$outdir/job_id/report.html" \
        -resume
else
    bash $BASEDIR/river/main.sh
fi