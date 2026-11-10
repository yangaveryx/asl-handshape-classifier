# Experiment notes

## Baseline

Evaluate full clips using image-coordinate Holistic landmarks, the original
27-node XY transform and shoulder normalization. The pretrained encoder and
handshape head are frozen in evaluation mode. All 50 outputs compete, including
`-1` (unannotated); scores are not signing-correctness percentages.

Initial exploratory clips included successful FAMILY/CLASS predictions and
incorrect WORKSHOP/GROUP predictions. Those anecdotes are not an accuracy study.
Local preprocessing comparisons matched the original single-video dataset path
on four synthetic cases, including missing detections. Raw overlays do not verify
normalization or checkpoint provenance. Transfer from WLASL to SEM-LEX remains
an experimental question.

## Pilot selection

Filter SEM-LEX metadata to ASL-LEX-labeled c/f/g/w candidates. Audit video IDs
against the full metadata, exclude split/annotation conflicts, collapse valid
repeated IDs and preserve alternative word labels. Seed 42 resolves ties in a
selection favoring least-used signers then word labels. Default targets are
20 train and 10 validation videos per category. Test rows are inventory only.
This is a diversity-oriented pilot, not a representative random sample.

The locally generated pilot had no train/validation video-ID overlap, but did
have signer overlap. It is not an unseen-signer evaluation. Generated lists are
not included here; the selection script records source hashes and exclusions.
Review labels and check video availability before interpreting evaluation.

## Next experiments

1. Run the fixed validation list to establish pretrained accuracy and coverage.
2. Inspect failures without deleting valid difficult clips.
3. Implement training separately; use training labels only for weight updates.
4. Compare on the same validation set; choose settings there.
5. Reserve untouched test data for final evaluation.

Do not train missing annotations as a physical handshape. Any future CL0/feature
score weighting is a separate validation experiment and is not implemented here.
