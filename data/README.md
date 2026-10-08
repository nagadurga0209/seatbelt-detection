# Dataset

The dataset is **not** committed to this repo (it's large and licensed by
its source). Download it into this folder before training.

## Expected layout (YOLO format)

```
data/
├── seatbelt/              # raw download (created by download_data.py)
│   ├── data.yaml
│   ├── train/
│   │   ├── images/*.jpg
│   │   └── labels/*.txt
│   ├── valid/
│   │   ├── images/*.jpg
│   │   └── labels/*.txt
│   └── test/              # (optional)
└── seatbelt_clahe/        # CLAHE-preprocessed copy (created by preprocess.py)
```

## Option A — Roboflow (recommended, scripted)

1. Make a free account at <https://roboflow.com> and copy your API key from
   <https://app.roboflow.com/settings/api>.
2. Add it to your `.env` as `ROBOFLOW_API_KEY=...`.
3. Run:
   ```bash
   python -m src.download_data
   ```

Default dataset: [`bassant/seatbeltfinal`](https://universe.roboflow.com/bassant/seatbeltfinal)
(~2,200 images, 2 classes). To use a different one, open its page on
[Roboflow Universe](https://universe.roboflow.com/search?q=seatbelt),
click **Download Dataset → YOLOv8 → show download code**, and pass the
values:
```bash
python -m src.download_data --workspace <ws> --project <proj> --version <n>
```

Other public options:
- [`seatbeltdet/seatbelt-vehicleoccupancy`](https://universe.roboflow.com/seatbeltdet/seatbelt-vehicleoccupancy)

## Option B — manual

Download any seat-belt detection dataset in **YOLOv8** format, unzip it into
`data/seatbelt/` following the layout above, and make sure its `data.yaml`
class names match `data/data.yaml` and `SB_VIOLATION_CLASSES`.

## After downloading

Always check the **class names** in the dataset's `data.yaml`. Update:
- `data/data.yaml` → `names:`
- `.env` / `src/config.py` → `SB_VIOLATION_CLASSES` (which class = violation)
