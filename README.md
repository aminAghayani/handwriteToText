# handwriteToText
persian/farsi handwrite to text with keras and data extraction

This project is trying to convert persian handwriting to text; For this purpose a dataset of 32 persian characters and 10 numbers are collected

Down below you can see one of the dataset sheets

<img src="https://user-images.githubusercontent.com/47675705/134969377-0deccf78-6a9e-4e34-a32d-e1bdf0aac9d6.jpg" width=20% height=50%>

Extract and classify datasets from these sheets is possible with aruco or corner detection

You can see some of these extracted characters and numbers down below

![pic1_10](https://user-images.githubusercontent.com/47675705/134947355-82918cca-3223-4371-9084-0952665bcb09.png)  ![pic4_7](https://user-images.githubusercontent.com/47675705/134947407-c31aa19e-74f1-4f15-9f9f-55d25964c725.png)  ![pic295_1](https://user-images.githubusercontent.com/47675705/134947605-9592e9f1-eba0-4a00-b00b-9105f575e691.png)  ![pic274_8](https://user-images.githubusercontent.com/47675705/134952721-ea57c389-2c23-4372-b981-d394a6559da2.png)

After training a CNN network with 28x28 input dimention and 42 outputs(persian characters + numbers) by extracted characters I reached about 90 percent of train and 88 percent of test accuracy

Better and wider dataset can help improve accuracy so much
