# Annotation Anonymization

The public reproducibility package uses stable anonymized annotator labels:

- `annotator_a`
- `annotator_b`
- `annotator_c`

The private identity mapping is intentionally not included in this repository.
The labels are stable across file names, CSV/XLSX column names, review tools,
and documentation.

## Naming Conventions

Annotation CSV files use:

```text
annotator_a_annotation.csv
annotator_b_annotation.csv
annotator_c_annotation.csv
```

Levenshtein-side source segmentation review files use:

```text
segmentation_annotator_a_*.csv
segmentation_annotator_b_*.csv
segmentation_annotator_c_*.csv
```

Annotator-specific columns use:

```text
Status_annotator_a
Correction_annotator_a
Status_annotator_b
Correction_annotator_b
Status_annotator_c
Correction_annotator_c
selected_type_annotator_a
selected_type_annotator_b
selected_type_annotator_c
```

## Workflow

The anonymized labels are only identifiers for reproducibility. They do not
change the annotation decisions, adjudication outcomes, final learner benchmark,
or reported paper results.
