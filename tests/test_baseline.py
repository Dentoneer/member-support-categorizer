from support_categorizer.baseline import predict_baseline, train_baseline

TEXTS = ["reset my password please", "forgot password reset link", "password not working",
         "where is my package shipping", "shipping is late package", "track my shipping box"]
LABELS = ["recover_password"] * 3 + ["status"] * 3


def test_baseline_learns_toy_labels():
    model = train_baseline(TEXTS, LABELS)
    assert predict_baseline(model, TEXTS) == LABELS
