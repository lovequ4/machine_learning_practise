# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers


# # 準備數據集
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     "img_data/train",
#     validation_split=0.2,
#     subset="training",
#     seed=123,
#     image_size=(224, 224),
#     batch_size=32,
#     label_mode="categorical"  # 將類別標籤轉換為 one-hot 編碼
# )
# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     "img_data/train",
#     validation_split=0.2,
#     subset="validation",
#     seed=123,
#     image_size=(224, 224),
#     batch_size=32,
#     label_mode="categorical"  # 將類別標籤轉換為 one-hot 編碼
# )
# print(train_ds.class_names)


# # 計算類別數量
# num_classes = len(train_ds.class_names)

# # 創建模型
# model = keras.Sequential(
#     [
#         layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(224, 224, 3)),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Flatten(),
#         layers.Dropout(0.5),
#         layers.Dense(num_classes, activation="softmax"),
#     ]
# )

# # 編譯模型
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# # 訓練模型
# model.fit(train_ds, validation_data=val_ds, epochs=20)

# model.save('fruit20.h5')

# #評估模型
# loss, acc = model.evaluate(val_ds)
# print("Loss:", loss)
# print("Accuracy:", acc)


 #--------------------------------------------------------------------------------------#

 

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

# # 準備數據集
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     "img_data/train",
#     validation_split=0.2,
#     subset="training",
#     seed=123,
#     image_size=(224, 224),
#     batch_size=32,
#     label_mode="categorical"  # 將類別標籤轉換為 one-hot 編碼
# )
# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     "img_data/train",
#     validation_split=0.2,
#     subset="validation",
#     seed=123,
#     image_size=(224, 224),
#     batch_size=32,
#     label_mode="categorical"  # 將類別標籤轉換為 one-hot 編碼
# )

# # 設置回調函數
# early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
# checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("fruit.h5", save_best_only=True)

# # 計算類別數量
# num_classes = len(train_ds.class_names)

# # 創建模型
# model = keras.Sequential(
#     [
#         layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(224, 224, 3)),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Flatten(),
#         layers.Dropout(0.5),
#         layers.Dense(num_classes, activation="softmax"),
#     ]
# )

# # 編譯模型
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# # 訓練模型
# history = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[early_stopping_cb, checkpoint_cb])

# # 載入最佳模型
# model = keras.models.load_model("fruit.h5")

# # 評估模型
# loss, acc = model.evaluate(val_ds)
# print("Loss:", loss)
# print("Accuracy:", acc)

# # 繪製訓練曲線
# import matplotlib.pyplot as plt
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()


#-----------------------------------------------------------
#ResNet50

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.applications.resnet50 import ResNet50

# # 準備數據集
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     "img_data/train",
#     validation_split=0.2,
#     subset="training",
#     seed=123,
#     image_size=(224, 224),
#     batch_size=32,
#     label_mode="categorical"  # 將類別標籤轉換為 one-hot 編碼
# )
# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     "img_data/train",
#     validation_split=0.2,
#     subset="validation",
#     seed=123,
#     image_size=(224, 224),
#     batch_size=32,
#     label_mode="categorical"  # 將類別標籤轉換為 one-hot 編碼
# )

# # 載入預訓練模型
# base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# # 凍結預訓練模型的權重
# base_model.trainable = False

# # 計算類別數量
# num_classes = len(train_ds.class_names)

# # 創建模型
# model = keras.Sequential(
#     [
#         base_model,
#         layers.GlobalAveragePooling2D(),
#         layers.Dropout(0.5),
#         layers.Dense(num_classes, activation="softmax"),
#     ]
# )

# # 編譯模型
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# # 訓練模型
# model.fit(train_ds, validation_data=val_ds, epochs=20)

# model.save('fruit_ResNet50.h5')

# #評估模型
# loss, acc = model.evaluate(val_ds)
# print("Loss:", loss)
# print("Accuracy:", acc)


#----------------------------------------------------------------------------------
 #MobileNetV2 預訓練模型
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 準備數據集
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "img_data/train",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    label_mode="categorical"  # 將類別標籤轉換為 one-hot 編碼
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "img_data/train",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    label_mode="categorical"  # 將類別標籤轉換為 one-hot 編碼
)
print(train_ds.class_names)

# 載入預訓練模型
base_model = keras.applications.MobileNetV2(
    weights="imagenet",  # 使用 ImageNet 預訓練權重
    include_top=False,  # 不包含頂層的全連接層
    input_shape=(224, 224, 3),
)

# 凍結所有預訓練模型層
for layer in base_model.layers:
    layer.trainable = False

# 增加自訂層
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(len(train_ds.class_names), activation="softmax")(x)
model = keras.Model(inputs=base_model.input, outputs=predictions)

# 編譯模型
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# 訓練模型
model.fit(train_ds, validation_data=val_ds, epochs=10)

# 解凍預訓練模型中的最後幾層
for layer in model.layers[:100]:
    layer.trainable = False
for layer in model.layers[100:]:
    layer.trainable = True

# 重新編譯模型
model.compile(
    optimizer=keras.optimizers.Adam(lr=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# 再次訓練模型，進行微調
model.fit(train_ds, validation_data=val_ds, epochs=20)

# 保存模型
model.save('fruit_MobileNetV2.h5')

#評估模型
loss, acc = model.evaluate(val_ds)
print("Loss:", loss)
print("Accuracy:", acc)
