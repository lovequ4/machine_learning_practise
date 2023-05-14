import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np

# 載入模型
model = keras.models.load_model("fruit_MobileNetV2.h5")

# 讀取圖片
img = cv2.imread("images2.jpg")

# 預處理圖片
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)

# 模型預測
preds = model.predict(img)[0]

# 取得預測結果最高的類別索引
pred_index = np.argmax(preds)

# 取得對應的類別名稱
class_names = ['apple', 'banana', 'cherry', 'chickoo', 'grapes', 'kiwi', 'mango', 'orange', 'strawberry']
fruit_name = class_names[pred_index]

# 繪製矩形框和水果名稱
font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.rectangle(img[0], (0, 0), (200, 40), (0, 0, 255), -1)
img = cv2.putText(img, fruit_name, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# 顯示結果圖片
cv2.imshow("Fruit", img)
cv2.waitKey(0)
cv2.destroyAllWindows()