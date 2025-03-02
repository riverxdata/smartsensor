.PHONY: process_image
process_image:
	smartsensor process \
		--data data/ampicilline/ip_1/raw_data \
		--outdir data/ampicilline/ip_1

modeling:
	smartsensor model \
		--data data/ampicilline/ip_1/features_rgb_raw_roi.csv \
		--meta data/ampicilline/ip_1/information.csv \
		--out data/ampicilline/ip_1
