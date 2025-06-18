.PHONY: process_image
process_image:
	smartsensor process \
		--data data/ampicilline/ip_1_10/raw_data \
		--outdir data/ampicilline/ip_1_10

modeling:
	smartsensor model \
		--degree 1 \
		--data data/ampicilline/ip_1_10/features_rgb_raw_roi.csv \
		--meta data/ampicilline/ip_1_10/information.csv \
		--out data/ampicilline/ip_1_10

modeling_ratio:
	smartsensor model \
		--norm ratio \
		--degree 1 \
		--data data/ampicilline/ip_1_10/features_rgb_raw_roi.csv \
		--meta data/ampicilline/ip_1_10/information.csv \
		--out data/ampicilline/ip_1_10_ratio

modeling_delta:
	smartsensor model \
		--norm delta \
		--data data/ampicilline/ip_1_10/features_rgb_raw_roi.csv \
		--meta data/ampicilline/ip_1_10/information.csv \
		--out data/ampicilline/ip_1_10_delta
