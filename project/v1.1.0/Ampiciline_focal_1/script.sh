# Define base paths
RAW_DATA=./data
OUTDIR=./outdir
KIT="1.1.0"

# Processing images
smartsensor process \
    --data "$RAW_DATA" \
    --kit $KIT \
    --auto-lum \
    --outdir "$OUTDIR/processed"

# Modeling 1 degree
smartsensor model \
    --data "$OUTDIR/processed" \
    --features "meanR,meanG,meanG,modeR,modeG,modeB" \
    --skip-feature-selection \
    --prefix raw \
    --out "$OUTDIR/model/raw" \
    --kit $KIT \
    --norm "raw" \
    --degree 1 \
    --replication 100 \
    --cv 5 \
    --test-size 0.2

smartsensor model \
    --data "$OUTDIR/processed" \
    --features "meanR,meanG,meanG,modeR,modeG,modeB" \
    --skip-feature-selection \
    --prefix delta \
    --out "$OUTDIR/model/delta" \
    --kit $KIT \
    --norm "delta" \
    --degree 1 \
    --replication 100 \
    --cv 5 \
    --test-size 0.2

smartsensor model \
    --data "$OUTDIR/processed" \
    --features "meanR,meanG,meanG,modeR,modeG,modeB" \
    --skip-feature-selection \
    --prefix ratio \
    --out "$OUTDIR/model/ratio" \
    --kit $KIT \
    --norm "ratio" \
    --degree 1 \
    --replication 100 \
    --cv 5 \
    --test-size 0.2


# Modeling 2 degree
smartsensor model \
    --data "$OUTDIR/processed" \
    --features "meanR,meanG,meanG,modeR,modeG,modeB" \
    --skip-feature-selection \
    --prefix raw \
    --out "$OUTDIR/model2/raw" \
    --kit $KIT \
    --norm "raw" \
    --degree 2 \
    --replication 100 \
    --cv 5 \
    --test-size 0.2

smartsensor model \
    --data "$OUTDIR/processed" \
    --features "meanR,meanG,meanG,modeR,modeG,modeB" \
    --skip-feature-selection \
    --prefix delta \
    --out "$OUTDIR/model2/delta" \
    --kit $KIT \
    --norm "delta" \
    --degree 2 \
    --replication 100 \
    --cv 5 \
    --test-size 0.2

smartsensor model \
    --data "$OUTDIR/processed" \
    --features "meanR,meanG,meanG,modeR,modeG,modeB" \
    --skip-feature-selection \
    --prefix ratio \
    --out "$OUTDIR/model2/ratio" \
    --kit $KIT \
    --norm "ratio" \
    --degree 2 \
    --replication 100 \
    --cv 5 \
    --test-size 0.2