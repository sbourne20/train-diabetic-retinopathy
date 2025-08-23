# Kaggle Dataset Download Guide

This guide provides two methods to download the diabetic retinopathy dataset directly from Kaggle to GCS, avoiding slow local uploads.

## Dataset Information

- **Kaggle Dataset**: https://www.kaggle.com/datasets/ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy/data
- **Target Folder**: `augmented_resized_V2` (we only want this folder, not `dr_unified_v2`)
- **Destination**: `gs://dr-data-2/dataset3_augmented_resized`

## Prerequisites

1. **Kaggle API Credentials**:
   - Go to https://www.kaggle.com/settings/account
   - Click "Create New API Token"
   - Download `kaggle.json` and note your username and key

2. **Google Cloud Setup**:
   - Ensure `dr-data-2` bucket exists
   - Have appropriate GCS permissions

## Method 1: Vertex AI Job (Recommended)

### Advantages
- Runs on Google Cloud infrastructure (fast download)
- Automatic monitoring and error handling
- Scalable and reliable

### Usage

```bash
# Run Kaggle download job on Vertex AI
python vertex_ai_kaggle_downloader.py \
  --action download \
  --kaggle_username YOUR_KAGGLE_USERNAME \
  --kaggle_key YOUR_KAGGLE_API_KEY \
  --bucket_name dr-data-2 \
  --project_id your-project-id \
  --target_folder dataset3_augmented_resized

# Monitor existing job
python vertex_ai_kaggle_downloader.py \
  --action monitor \
  --job_id projects/.../jobs/... \
  --bucket_name dr-data-2 \
  --project_id your-project-id
```

### What it does:
1. Creates a Vertex AI Custom Job with CPU machine
2. Installs Kaggle API in the job
3. Downloads dataset directly to job storage
4. Extracts only `augmented_resized_V2` folder
5. Uploads directly to GCS
6. Verifies upload structure

## Method 2: Colab Enterprise (Alternative)

### Advantages
- Interactive environment
- Easy to run and monitor
- Good for one-time downloads

### Usage in Colab Enterprise

1. **Upload the script**:
   ```python
   # Upload colab_kaggle_downloader.py to Colab
   ```

2. **Run interactively**:
   ```python
   # In Colab cell
   exec(open('colab_kaggle_downloader.py').read())
   ```

3. **Or run with parameters**:
   ```python
   from colab_kaggle_downloader import download_kaggle_dataset_to_gcs
   
   download_kaggle_dataset_to_gcs(
       kaggle_username="your_username",
       kaggle_key="your_api_key", 
       bucket_name="dr-data-2",
       target_folder="dataset3_augmented_resized"
   )
   ```

### Command Line Usage
```bash
python colab_kaggle_downloader.py \
  --kaggle_username YOUR_KAGGLE_USERNAME \
  --kaggle_key YOUR_KAGGLE_API_KEY \
  --bucket_name dr-data-2 \
  --target_folder dataset3_augmented_resized
```

## Expected Results

After successful download, you'll have:

```
gs://dr-data-2/dataset3_augmented_resized/
├── train/
│   ├── 0/     (~55,162 images - No DR)
│   ├── 1/     (~18,470 images - Mild NPDR)  
│   ├── 2/     (~24,198 images - Moderate NPDR)
│   ├── 3/     (~7,936 images - Severe NPDR)
│   └── 4/     (~9,475 images - PDR)
├── val/
│   ├── 0/     (~6,895 images)
│   ├── 1/     (~1,840 images)
│   ├── 2/     (~3,024 images)
│   ├── 3/     (~1,000 images)
│   └── 4/     (~1,468 images)
└── test/
    ├── 0/     (~6,896 images)
    ├── 1/     (~1,862 images)
    ├── 2/     (~2,999 images)
    ├── 3/     (~978 images)
    └── 4/     (~1,466 images)
```

**Total**: ~143,669 images (115k training, 14k validation, 14k test)

## Verification

Both scripts include verification steps that will:
1. Check all required directories exist
2. Count files per class and split
3. Verify expected file counts
4. Report any missing directories or unexpected counts

## After Download

Once download is complete, you can train with:

```bash
python vertex_ai_trainer.py \
  --action train \
  --dataset dataset3_augmented_resized \
  --dataset-type 1 \
  --bucket_name dr-data-2 \
  --project_id your-project-id
```

## Troubleshooting

### Common Issues

1. **Kaggle API Authentication Error**:
   - Verify username and API key are correct
   - Ensure you've accepted the dataset's terms on Kaggle

2. **GCS Permission Error**:
   - Verify you have Storage Admin or equivalent permissions
   - Check that the bucket exists

3. **Vertex AI Job Fails**:
   - Check job logs in Google Cloud Console
   - Verify project has Vertex AI API enabled

4. **Incomplete Download**:
   - Check available disk space
   - Monitor GCS storage quotas
   - Re-run the script (it will skip existing files)

### File Count Verification

Expected approximate counts:
- **Training**: 115,241 files
- **Validation**: 14,227 files  
- **Test**: 14,201 files
- **Total**: 143,669 files

If counts are significantly different, the download may have been interrupted.

## Performance

- **Vertex AI**: ~30-60 minutes for full download
- **Colab Enterprise**: ~45-90 minutes depending on instance
- **Network Transfer**: Much faster than local upload (10-50x speedup)

## Cost Considerations

- **Vertex AI Job**: ~$2-5 for n1-standard-4 running 1 hour
- **Colab Enterprise**: Included in Colab compute units
- **GCS Storage**: ~$0.50/month for 150GB dataset storage