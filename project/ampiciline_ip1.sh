# Define base paths
BASE_DIR="data/ampicilline/ip_1_10"
RAW_DATA="$BASE_DIR/raw_data"
FEATURES="$BASE_DIR/features_rgb_raw_roi.csv"
META="$BASE_DIR/information.csv"

# Processing images
smartsensor process \
    --data "$RAW_DATA" \
    --outdir "$BASE_DIR" > "$BASE_DIR/sensor.log"

# Create directories for modeling outputs
for norm in raw ratio delta; do
    mkdir -p "${BASE_DIR}_${norm}"
done

# Modeling
smartsensor model \
    --data "$FEATURES" \
    --degree 2 \
    --meta "$META" \
    --out "${BASE_DIR}_raw" > "${BASE_DIR}_raw/sensor.log"

smartsensor model \
    --norm ratio \
    --data "$FEATURES" \
    --degree 2 \
    --meta "$META" \
    --out "${BASE_DIR}_ratio" > "${BASE_DIR}_ratio/sensor.log"

smartsensor model \
    --norm delta \
    --data "$FEATURES" \
    --degree 2 \
    --meta "$META" \
    --out "${BASE_DIR}_delta" > "${BASE_DIR}_delta/sensor.log"
