# Cadastral data

* Download Norwegian cadastral here: [Matrikkelen - Eiendomskart Teig (GeoNorge)](https://kartkatalog.geonorge.no/metadata/kartverket/matrikkelen-eiendomskart-teig/74340c24-1c8a-4454-b813-bfe498e80f16). Pick `GML` as the output format.
* Convert file to `ESRI Shapefile` by executing `ogr2ogr -f "ESRI Shapefile" <downloaded_file> <output_directory>`.
* The cadastral data is now available in `<output_directory>/Teig.shp`.

# Building data

```
ogr2ogr -f "GPKG" buildings.gpkg Basisdata_5001_Trondheim_5972_FKB-Bygning_GML.gml
```

# LiDAR data

```
gdalbuildvrt lidar.vrt <unzipped_data>/data/dom/*.tif
```
