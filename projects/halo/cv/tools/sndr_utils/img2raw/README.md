## img2raw.py
- 作用：从dataset中读取图片并保存成raw格式。
- 用法：
```
python3 img2raw.py --config 'config' --interval 100 --sample-number 100 --saved-path './saved_img'
```
- config：配置文件，可参考config.py配置自己的文件，主要需要dataset,dataset list,transform,dataloader。
- interval: 图片采样间隔
- sample-number: 图片采样的总数
- saved-path: 保存图片的文件夹名