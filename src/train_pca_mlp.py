from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "ISTE-HAR_dataset.csv"
MODEL_DIR = ROOT_DIR / "models"
RANDOM_STATE = 8


def load_data():
    df = pd.read_csv(DATA_PATH)
    x = df.drop("label", axis=1).values
    y = df["label"].values
    return df, x, y


def build_model(input_dim: int):
    model = Sequential(
        [
            Dense(
                48,
                input_shape=(input_dim,),
                activation="relu",
                kernel_regularizer=l2(0.008),
            ),
            BatchNormalization(),
            Dropout(0.3),
            Dense(24, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.003),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    df, x, y = load_data()
    print(f"Veri seti boyutu: {df.shape}")
    print("Sınıf dağılımı:")
    print(df["label"].value_counts())

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    pca = PCA(n_components=15)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)

    print(f"PCA sonrası eğitim verisi: {x_train_pca.shape}")
    print(f"PCA sonrası test verisi: {x_test_pca.shape}")

    model = build_model(input_dim=x_train_pca.shape[1])
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=25,
        restore_best_weights=True,
        verbose=0,
    )

    model.fit(
        x_train_pca,
        y_train,
        epochs=200,
        batch_size=16,
        validation_split=0.15,
        callbacks=[early_stopping],
        verbose=1,
    )

    train_pred = (model.predict(x_train_pca, verbose=0) > 0.5).astype(int).flatten()
    test_pred = (model.predict(x_test_pca, verbose=0) > 0.5).astype(int).flatten()

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"Train accuracy: {train_acc * 100:.2f}%")
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, test_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, test_pred))

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "iste_har_mlp_pca_model.keras"
    model.save(model_path)
    print(f"Model kaydedildi: {model_path}")


if __name__ == "__main__":
    main()
