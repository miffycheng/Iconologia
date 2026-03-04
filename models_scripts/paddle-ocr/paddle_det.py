from paddleocr import TextDetection
model = TextDetection(model_name="PP-OCRv5_mobile_det")
output = model.predict(input="./data/Amsterdam_1698_Jean_Baudoin/selected/page_31.jpg", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res_det.json")
