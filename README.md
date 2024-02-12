# 关于Catboost底层源码的实现分析

[Website](https://catboost.ai) |
[Documentation](https://catboost.ai/docs/) |
[Tutorials](https://catboost.ai/docs/concepts/tutorials.html) |
[Installation](https://catboost.ai/docs/concepts/installation.html) |
[Release Notes](https://github.com/catboost/catboost/releases)

[![Twitter](https://img.shields.io/badge/@CatBoostML--_.svg?style=social&logo=twitter)](https://twitter.com/CatBoostML)

`说明:`[catboost源码定期会更新,下述代码基于2024.02.06master源码]
----
### 分析目的: 主要是查看 `底层BuildTree的实现` 和 `对应文献中解决预测偏差创建的模型M` 以及 `余弦相似度的实现`

* `代码的具体接口使用见CatboostModelAPI.md说明`
- CatBoost 是一种支持多线程的机器学习算法库,CatBoost 使用了多线程来加速模型训练过程，尤其是在处理大量数据和复杂模型时能够发挥其优势。
----

- 首先从github上获取catboost的源码,catboost的底层实现是c++,所以当使用python运行catboost时,相当于使用提供的python接口,对训练的数据和配置的超参数进行解析,并传入c++底层代码中进行运行。
- 对克隆下来的[https://github.com/catboost/catboost]文件打开其中的catboost文件夹,源码在此处。
- `./python-package/catboost/core.py`文件中定义了在使用python应用catboost时所调用的' `CatBoostRegressor`和`CatBoostClassifier类`,其都为_CatBoostBase的父类的子类进行继承,前两者的上层为CatBoost类,内部的_fit()方法,是对数据的训练。
- 在_CatBoostBase类的构造函数中的初始化中出现:
``` 
## code core.py
def __init__(self, params):
init_params = params.copy() if params is not None else {}
stringify_builtin_metrics(init_params)
self._init_params = init_params
if 'thread_count' in self._init_params and self._init_params['thread_count'] == -1:
    self._init_params.pop('thread_count')
if 'fixed_binary_splits' in self._init_params and self._init_params['fixed_binary_splits'] is None:
    self._init_params['fixed_binary_splits'] = []
self._object = _CatBoost()
```
最后一行的 `self._object = _CatBoost()` 则是调用_catboost文件内的内容
```
## code core.py
from . import _catboost
from .metrics import BuiltinMetric


_typeof = type

_PoolBase = _catboost._PoolBase
_CatBoost = _catboost._CatBoost
```
- `_catboost`文件为_catboost.pyx(与core.py属于统一文件夹内)
```
cdef extern from "catboost/libs/train_lib/train_model.h":
    cdef void TrainModel(
        TJsonValue params,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        const TMaybe[TCustomObjectiveDescriptor]& objectiveDescriptor,
        const TMaybe[TCustomMetricDescriptor]& evalMetricDescriptor,
        const TMaybe[TCustomCallbackDescriptor]& callbackDescriptor,
        TDataProviders pools,
        TMaybe[TFullModel*] initModel,
        THolder[TLearnProgress]* initLearnProgress,
        const TString& outputModelPath,
        TFullModel* dstModel,
        const TVector[TEvalResult*]& testApproxes,
        TMetricsAndTimeLeftHistory* metricsAndTimeHistory,
        THolder[TLearnProgress]* dstLearnProgress
    ) nogil except +ProcessException
```
可知调用的是`"catboost/libs/train_lib/train_model.h"`文件中的`TrainModel`函数,声明一个名为TrainModel的外部函数,
该函数在C/C++的头文件"catboost/libs/train_lib/train_model.h"中定义。
在train_model.cpp文件中定义了多个TrainModel()函数,但是传入参数不一致,`调用的过程如下:`

- 首先调用1607行的train_model.cpp文件最后的TrainModel函数(输入参数为13个),对来自python文件的参数进行预处理。
- 在第一个调用的TrainModel内的最后调用函数946行的TrainModel(输入参数为17个):
- 该函数执行内容: `1.更新输出选项` `2.判断CPU和GPU的选择` `3.配置catBoostOptions()` `4.提取损失函数` `5.量化特征信息` `6.ReorderByTimestampLearnDataIfNeeded()如果需要按照时间戳对数据进行重排列` `7.ShuffleLearnDataIfNeeded()如果需要重新打乱学习数据` `8.分析内存` `9.进行标签转化` `10.最后调用modelTrainerHolder->TrainModel()函数 (1154行)`
- modelTrainerHolder->TrainModel()函数位于773行(输入参数为19个):
- 该函数执行内容: `1.参数初始化` `2.定义初始的近似值Approx` `3.创建TLearnContext ctx变量用于存储特定学习数据结构的类,例如prng、学习进度和目标分类器` `4.调用Train()函数位于375行` `5.然后保存模型SaveModel()并对模型进行评估ModelBasedEval()`
- 调用的Train()函数,该函数执行内容如下:
```
    ##375行 Train函数下的419行迭代循环
    ...
    InitializeAndCheckMetricData(internalOptions, data, *ctx, &metricsData);
    ...
    for (ui32 iter = ctx->LearnProgress->GetCurrentTrainingIterationCount();
         continueTraining && (iter < ctx->Params.BoostingOptions->IterationCount);
         ++iter)
    {
        ...
        profile.StartNextIteration();
        ...
        TrainOneIteration(data, ctx);
        ...
        profile.FinishIteration();
        ...
    }
    ...
    ctx->SaveProgress(onSaveSnapshotCallback);
    ...
```
由上述循环可知,在迭代循环内部调用TrainOneIteration(data, ctx);函数对训练数据进行第一迭代,大部分函数调用`./catboost/private/libs/algo/`目录下的算法函数文件,例如其中的`train.cpp`
```
void TrainOneIteration(const NCB::TTrainingDataProviders& data, TLearnContext* ctx) {
    ...
    CheckInterrupted(); // check after long-lasting operation
    #创建最佳树变量
    std::variant<TSplitTree, TNonSymmetricTreeStructure> bestTree;
    {
        ... 
        #采用贪婪方式进行搜索
        GreedyTensorSearch(
        data,
        modelLength,
        profile,
        takenFold,
        ctx,
        &bestTree
        );
    }
    ...
    CheckInterrupted(); // check after long-lasting operation
    {
        ... #计算树状结构的在线CTR
    }
    CheckInterrupted(); // check after long-lasting operation

    TVector<TVector<double>> treeValues; // [dim][leafId]
    TVector<double> sumLeafWeights; // [leafId]
    ...
    #近似计算叶子结果
    if (
        ctx->Params.ObliviousTreeOptions->DevLeafwiseApproxes.Get() &&
        ctx->Params.BoostingOptions->BoostingType.Get() == EBoostingType::Plain
        && !treeHasMonotonicConstraints
        && error->GetErrorType() == EErrorType::PerObjectError
    ) {
        CalcApproxesLeafwise(
            data,
            *error,
            bestTree,
            ctx,
            &treeValues,
            &indices
        );
    } else {
        CalcLeafValues(
            data,
            *error,
            bestTree,
            ctx,
            &treeValues,
            &indices
        );
    }
    ...
    #近似计算树结构和更新树结构近似值
    CheckInterrupted(); // check after long-lasting operation
    TConstArrayRef<ui32> learnPermutationRef = ctx->LearnProgress->AveragingFold.GetLearnPermutationArray();

    const size_t leafCount = treeValues[0].size();
    sumLeafWeights = SumLeafWeights(
        leafCount,
        indices,
        learnPermutationRef,
        GetWeights(*data.Learn->TargetData),
        ctx->LocalExecutor
    );
    const auto lossFunction = ctx->Params.LossFunctionDescription->GetLossFunction();
    const bool usePairs = UsesPairsForCalculation(lossFunction);
    NormalizeLeafValues(
        usePairs,
        ctx->Params.BoostingOptions->LearningRate,
        sumLeafWeights,
        &treeValues
    );
    ...
    #更新最近的近似
    ctx->LearnProgress->TreeStats.emplace_back();
    ctx->LearnProgress->TreeStats.back().LeafWeightsSum = std::move(sumLeafWeights);
    ctx->LearnProgress->LeafValues.push_back(std::move(treeValues));
    ctx->LearnProgress->TreeStruct.push_back(std::move(bestTree));
    ...
    CheckInterrupted(); // check after long-lasting operation
}
```
