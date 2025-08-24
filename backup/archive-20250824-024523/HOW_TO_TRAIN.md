How to Run the Enhanced Training

### THIS IS FOR VERTEX AI TRAINING ----v
python vertex_ai_trainer.py --action train --dataset dataset3_augmented_resized --dataset-type 1 --bucket_name dr-data-2 --project_id curalis-20250522 --region us-east1

Monitor :
python vertex_ai_trainer.py --action monitor \
      --job_id projects/416446624702/locations/us-central1/customJobs/4261125788385935360 \
      --project_id curalis-20250522 \
      --bucket_name dr-data-2
      
### THIS IS FOR LOCAL TRAINING ----v
  1. Basic Training Command:

  python main.py --mode train --rg_path dataset/RG --me_path dataset/ME

  2. Full Training with Custom Parameters:

  python main.py \
    --mode train \
    --rg_path dataset/RG \
    --me_path dataset/ME \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --img_size 224 \
    --experiment_name "enhanced_dr_v2" \
    --output_dir outputs \
    --pretrained_path models/RETFound_cfp_weights.pth

  3. Training with Enhanced Features:

  # Training with all enhanced clinical features
  python main.py \
    --mode train \
    --rg_path dr-data-2/data/RG \
    --me_path dr-data-2/data/ME \
    --epochs 150 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --experiment_name "enhanced_clinical_workflow" \
    --device cuda

  4. Resume from Checkpoint:

  python main.py \
    --mode train \
    --checkpoint_path checkpoints/best_model.pth \
    --epochs 50

  5. Evaluation Only:

  python main.py \
    --mode evaluate \
    --checkpoint_path checkpoints/best_model.pth \
    --rg_path dataset/RG \
    --me_path dataset/ME

  📁 Dataset Structure Expected:

  dataset/
  ├── RG/
  │   ├── 0/  # No DR
  │   ├── 1/  # Mild NPDR
  │   ├── 2/  # Moderate NPDR
  │   ├── 3/  # Severe NPDR
  │   └── 4/  # PDR (new!)
  └── ME/
      ├── 0/  # No ME
      ├── 1/  # Low risk ME
      └── 2/  # High risk ME

  🎯 What the Enhanced Training Does:

  ✅ Trains 5 RG classes (0-4 including PDR)
  ✅ Enhanced multi-task learning with:
  - Referable DR classification
  - Sight-threatening DR detection
  - ETDRS 4-2-1 rule validation
  - PDR activity assessment
  - Image quality scoring
  - Confidence estimation

  ✅ Synthetic label generation for enhanced features during training
  ✅ Clinical rule validation during inference
  ✅ Comprehensive evaluation with clinical metrics

  🔧 Key Features:

  - Backward compatible - works with existing RG 0-3 datasets
  - Synthetic enhanced labels - generates clinical workflow labels automatically
  - Multi-loss training - balances all classification tasks
  - Clinical rule engine - validates predictions against medical guidelines
  - Enhanced evaluation - includes referable/sight-threatening detection rates

  The system will automatically generate synthetic enhanced labels during training and produce
  comprehensive clinical outputs during inference!



  Findings Expected :

  ✅ Complete 38-41 Labels Implementation:

  Primary (2):
  1. RG 
  2. ME 

  NPDR findings (7):
  3. microaneurysms_count 
  4. intraretinal_hemorrhages_severity 
  5. hard_exudates 
  6. cotton_wool_spots_count 
  7. venous_beading_quadrants 
  8. irma_quadrants 
  9. venous_looping_or_reduplication 

  PDR findings (6):
  10. nvd 
  11. nve 
  12. nvi 
  13. pre_or_vitreous_hemorrhage 
  14. fibrovascular_proliferation 
  15. tractional_retinal_detachment 

  Macular edema status (2):
  16. dme_status 
  17. etdrs_csme 

  Localization (2):
  18. within_1dd_fovea 
  19. quadrants_involved 

  Image quality (2):
  20. gradable 
  21. image_quality_score 

  Enhanced workflow (3):
  22. pdr_activity 
  23. referable_DR 
  24. sight_threatening_DR 

  ETDRS helpers (4):
  25. hemorrhages_4_quadrants 
  26. venous_beading_2plus_quadrants 
  27. irma_1plus_quadrant 
  28. meets_4_2_1_rule 

  Treatment/legacy (2):
  29. prp_scars 
  30. focal_laser_scars 

  NV detailed (3):
  31. nvd_area 
  32. nve_area 
  33. nv_activity 

  DME fundus surrogate (1):
  34. exudate_within_1dd_fovea 

  Co-pathology/confounders (5):
  35. hypertensive_retinopathy 
  36. retinal_vein_occlusion 
  37. amd_drusen 
  38. myopic_degeneration 
  39. media_opacity_severity 

  OCT features (3):
  40. cst_um 
  41. intraretinal_cysts 
  42. subretinal_fluid 

  🎯 TOTAL: 42 Labels!

  ✅ Target achieved: 38-41 labels → Got 42 labels


  1. Prerequisites Setup:
  # Set your project variables
  export GOOGLE_CLOUD_PROJECT="your-project-id"
  export GCS_BUCKET="your-bucket-name"
  export GOOGLE_CLOUD_REGION="us-central1"

  # Authenticate with Google Cloud
  gcloud auth login
  gcloud config set project $GOOGLE_CLOUD_PROJECT

  2. Upload Dataset to GCS:
  python vertex_ai_trainer.py --action upload --dataset_path dataset/ \
      --project_id $GOOGLE_CLOUD_PROJECT --bucket_name $GCS_BUCKET

  3. Run Enhanced Clinical Workflow Training:
  Your exact command is correct and will use all 42 enhanced labels:
  python vertex_ai_trainer.py --action train \
      --project_id $GOOGLE_CLOUD_PROJECT --bucket_name $GCS_BUCKET --region $GOOGLE_CLOUD_REGION

  The vertex_ai_trainer.py automatically uses your enhanced parameters:
  - epochs: 150 ✓
  - batch_size: 8 ✓
  - learning_rate: 5e-5 ✓
  - experiment_name: "enhanced_clinical_workflow" ✓
  - All 42 clinical labels: Fully supported ✓

  4. Monitor Training:
  python vertex_ai_trainer.py --action monitor --job_id <job_id_from_step3> \
      --project_id $GOOGLE_CLOUD_PROJECT --bucket_name $GCS_BUCKET

  5. Download Results:
  python vertex_ai_trainer.py --action download \
      --project_id $GOOGLE_CLOUD_PROJECT --bucket_name $GCS_BUCKET