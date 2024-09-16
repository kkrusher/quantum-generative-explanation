

generative_models.py: 定义生成模型
generative_trainer.py: 生成模型的训练器   
classification_models.py: 定义分类器           
classification_trainer.py: 分类器的训练器

utils.py: 功能性函数       
loss.py: 定义了损失函数，如MSE，cross entropy, fidelity distance                          

data: 生成的数据集
results: 实验结果

generate_SPTS_dataset.py: 生成SPTS数据集
generate_TFIM_dataset.py: 生成TFIM数据集 

report_results.py: ipynb文件中共同用到的一些变量，用于指定当前的实验配置等

对一组实验，我们按照以下流程进行实验：
- 在report_results中指定实验配置
- 运行 train_generative_model.ipynb 训练生成器
- 运行 train_classification_model.ipynb 训练分类器
- 使用 explain_control.ipynb 对训练好的模型进行可解释性控制
- 使用 plot-loss.ipynb 打印训练过程中的损失函数降低曲线 



