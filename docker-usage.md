# PDV Docker Usage

This Docker image provides a containerized version of the PDV (Proteomics Data Viewer) command-line interface with virtual display support.

## Building the Image

```bash
docker build -t pdv-cli .
```

## Technical Details

- **Virtual Display**: Uses Xvfb (X Virtual Framebuffer) to handle PDV's GUI components in headless mode
- **Base Image**: OpenJDK 8 JRE Alpine Linux
- **Memory**: Recommend 4-8GB+ for large datasets (>1GB files)

## Usage

### Show Help
```bash
docker run --rm pdv-cli
```

### Basic Usage Example
```bash
# Mount your data directory and run PDV CLI
docker run --rm --memory=4g \
  -v /path/to/your/data:/data \
  pdv-cli \
  -rt 1 \
  -r /data/results.mzid \
  -st 1 \
  -s /data/spectra.mgf \
  -k s \
  -i /data/spectrum_ids.txt \
  -o /data/output \
  -ft png
```

### Parameters Explanation

**Required Parameters:**
- `-rt`: Result file type (1=mzIdentML, 2=pepXML, 3=proBAM, 4=txt, 5=maxQuant, 6=TIC)
- `-r`: Identification file path
- `-st`: Spectrum file type (1=mgf, 2=mzML, 3=mzXML)
- `-s`: MS/MS data file path
- `-i`: File containing peptide sequences or spectrum IDs to process
- `-k`: Input data type (s=spectrum ID, p=peptide sequence)
- `-o`: Output directory
- `-ft`: Figure type (png, pdf, tiff, report)

**Optional Parameters:**
- `-a`: Error window for fragment ion mass values (default: 0.5 Da)
- `-c`: Intensity percentile for annotation (default: 3%)
- `-fh`: Figure height (default: 400)
- `-fw`: Figure width (default: 800)
- `-fu`: Units for height/width (cm, mm, px - default: px)
- `-ah`: Consider neutral loss of H2O
- `-an`: Consider neutral loss of NH3
- `-rp`: Remove precursor peak

### Creating Input Files

PDV requires a file listing which spectra to process. You can create this from your identification file:

#### From mzIdentML files:
```bash
# Extract spectrum IDs from mzid file
docker run --rm --entrypoint sh -v /path/to/data:/data pdv-cli -c \
  "grep -o 'spectrumID=\"index=[0-9]*\"' /data/your_file.mzid | \
   sed 's/spectrumID=\"//' | sed 's/\"//' | head -10 > /data/spectrum_ids.txt"
```

#### Manual creation:
```bash
# Create a simple text file with spectrum IDs (one per line)
echo "index=12345" > spectrum_ids.txt
echo "index=67890" >> spectrum_ids.txt
```

### Volume Mounts

- Mount your input data to `/data` inside the container
- Output files will be created in the specified output directory within your mounted volume

## Example Data Structure

```
/your/data/
├── results.mzid          # Identification results
├── spectra.mgf           # MS/MS spectra
├── spectrum_ids.txt      # List of spectrum IDs to process
└── output/               # Output directory (will be created)
    ├── spectrum1.png
    ├── spectrum2.png
    └── ...
```

## Memory Requirements

For large datasets, allocate sufficient memory:

```bash
# For datasets >1GB, use 4-8GB memory
docker run --rm --memory=8g \
  -v /path/to/data:/data \
  pdv-cli [parameters...]

# For very large datasets (>5GB), consider:
docker run --rm --memory=16g \
  -e JAVA_OPTS="-Xmx12g -Xms4g" \
  -v /path/to/data:/data \
  pdv-cli [parameters...]
```

## Troubleshooting

### Memory Issues
**Error**: `library initialization failed - unable to allocate file descriptor table - out of memory`

**Solutions:**
1. Increase Docker memory allocation: `--memory=8g` or higher
2. Process smaller subsets of data
3. Ensure Docker Desktop has sufficient memory allocated (Settings > Resources)
4. Consider using smaller input files for testing first

### File Not Found Errors
- Ensure your data directory is properly mounted with `-v`
- Check file paths are correct inside the container (`/data/...`)
- Verify file permissions allow Docker to read them

### Display Issues
The container uses Xvfb for virtual display - no additional configuration needed for headless operation.

## TIC Mode Example

For Total Ion Chromatogram generation (doesn't require spectrum IDs):

```bash
docker run --rm --memory=4g \
  -v /path/to/data:/data \
  pdv-cli \
  -rt 6 \
  -r /data/results.mzid \
  -st 1 \
  -s /data/spectra.mgf \
  -o /data/output \
  -ft png
```
