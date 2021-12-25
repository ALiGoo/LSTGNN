wget -P raw_data_201306_201709 https://gbfs.citibikenyc.com/gbfs/en/station_information.json
cat raw_data_urls_201307_201709.txt | xargs -n 1 -P 6 wget -P raw_data_201306_201709/
unzip 'raw_data_201306_201709/*.zip' -d raw_data_201306_201709/