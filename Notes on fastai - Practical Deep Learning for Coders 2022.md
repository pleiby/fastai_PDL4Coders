Notes on fastai - Practical Deep Learning for Coders 2022
==========================================================

[Lesson 1: Getting started](https://course.fast.ai/Lessons/lesson1.html)
------------------------------------------------------------------------

- Pretrained Models and Transfer Learning
    - Pretrained models are useful because they have already learned how to [identify and] handle a lot of simple features like edge and color detection. 
        - However, since the model was trained for a different task than already used, this model cannot be used as is.
    - `vision_learner` also has a parameter `pretrained`, which defaults to `True`
    - When using a pretrained model, `vision_learner` will remove the last layer, since that is always specifically customized to the original training task (i.e. ImageNet dataset classification), and replace it with one or more new layers with randomized weights, of an appropriate size for the dataset you are working with. This last part of the model is known as the _head_.
    - Using a pretrained model for a task different to what it was originally trained for is known as **transfer learning**. 
        - Unfortunately, because transfer learning is so under-studied, few domains have pretrained models available.
        - in addition, it is not yet well understood how to use transfer learning for tasks such as time series analysis.
    - there are some important tricks to adapt a pretrained model for a new dataset—a process called fine-tuning.

- Image Recognizers Can Tackle Non-Image Tasks
    - An image recognizer can, as its name suggests, only recognize images. But a lot of things can be represented as images, which means that an image recogniser can learn to complete many tasks.
    - This [result from image-based malware detection] suggests a good rule of thumb for converting a dataset into an image representation: if the human eye can recognize categories from the images, then a deep learning model should be able to do so too.

- In general, you'll find that a small number of general approaches in deep learning can go a long way, if you're a bit creative in how you represent your data! 

- When we train a model, a key concern is to ensure that our model generalizes—that is, that it learns general lessons from our data which also apply to new items it will encounter .... In order to avoid this [_overfitting_], we always divide our data into two parts, the _training set_ and the _validation set_. 
- During the training process, when the model has seen every item in the training set, we call that an _epoch._ [which may involve many batches]

- What makes deep learning distinctive is a particular class of architectures ... based on neural networks. 
    - In particular, tasks like image classification rely heavily on _convolutional neural networks_
- Deep Learning Is Not Just for Image Classification: other applications
    - Creating a model that can recognize the content of every individual pixel in an image is called _segmentation_
    - natural language processing (NLP)
        - sentiment classification, generate text, translate automatically from one language to another, analyze comments, label words in sentences
    -  building models from plain tabular data.
        - _Tabular_: Data that is in the form of a table, such as from a spreadsheet, database, or CSV file. A tabular model is a model that tries to predict one column of a table based on information in other columns of the table.
        - in general, pretrained models are not widely available for any tabular modeling tasks, although some organizations have created them for internal use
            - so instead of using _fine_tune_ in the call to _learn_, we use _fit_one_cycle_, the most commonly used method for training fastai models from scratch
    - Recommendation systems (Collaborative Learning)
        - not actually using a pretrained model (for the same reason that we didn't for the tabular model), but fastai lets us use _fine_tune_ anyway in this case 

- Validation Sets and Test Sets
    - first step [to support generilizability of results] was to split our dataset into two sets: the training set (which our model sees in training) and the validation set, also known as the development set (which is used only for evaluation). 
    - in realistic scenarios we rarely build a model just by training its weight parameters once. Instead, we are likely to explore many versions ... [with variations] regarding network architecture, learning rates, data augmentation strategies (_hyperparameters_)
    - We evaluate the model by looking at predictions on the _validation_ data when we explore new hyperparameter values
        - we are in danger of overfitting the validation data through human trial and error and exploration
    - solution to this conundrum is to introduce another level of even more highly reserved data, the _test set_. 
        - hold back the validation data from the training process
        - "we must hold back the test set data even from ourselves. It cannot be used to improve the model; it can only be used to evaluate the model at the very end of our efforts"
    - Then you check their model on your test data, using a metric that you choose based on what actually matters to you in practice, and you decide what level of performance is adequate.
    - (It's also a good idea for you to try out some simple baseline yourself, so you know what a really simple model can achieve [in comparison to the developed model].
    - Use Judgment in Defining Test Sets
        - a key property of the validation and test sets is that they must be representative of the new data you will see in the future
            - Sometimes this isn’t true if a random sample is used.
        - for time series: "If your data includes the date and you are building a model to use in the future, you will want to choose a continuous section with the latest dates as your validation set"
            - use the earlier data as your training set (and the later data for the validation set)
        - [when possible] anticipate ways the data you will be making predictions for in production may be qualitatively different from the data you have to train your model with 
            - [e.g., for panel data, validation/test data might be on categories/individuals it hasn't seen before in training]
            - ? Note: this seems the opposite of using stratified sampling for validation/test set with panel data

- Key things to consider when using AI in an organization: (summary)
    - Make sure a training, validation, and testing set is defined properly in order to evaluate the model in an appropriate manner.
    - Try out a simple baseline, which future models should hopefully beat. Or even this simple baseline may be enough in some cases.

-  the [_universal approximation theorem_](https://en.wikipedia.org/wiki/Universal_approximation_theorem#History) states that neural networks (with at least one hidden layer) can theoretically represent any mathematical function.

#### Jupyter Notebook-bsed Applications
- Presentations: [RISE](https://rise.readthedocs.io/en/stable/)
- Blogging: [fastpages](https://github.com/fastai/fastpages)
- The notebooks used to create the [fastai library](https://github.com/fastai/fastai/tree/master/nbs)
- [nbdev](https://nbdev.fast.ai/) - the system we built to create Python libraries using Jupyter and CI
- Jupyter Notebook Extensions - allows Table of Contents, Navigating sections
#### Beginner Help
- [Help: Setup](https://forums.fast.ai/t/help-setup/95289)
- [Help: Creating a dataset, and using Gradio / Spaces](https://forums.fast.ai/t/help-creating-a-dataset-and-using-gradio-spaces/96281)
- [Help: Using Colab or Kaggle](https://forums.fast.ai/t/help-using-colab-or-kaggle/96280)

[Lesson 2: Deployment](https://course.fast.ai/Lessons/lesson2.html)
------------------------------------------------------------------------

- There are many accurate models that are of no use to anyone, and many inaccurate models that are highly useful. To ensure that your modeling work is useful in practice, you need to consider how your work will be used. In 2012 Jeremy, along with Margit Zwemer and Mike Loukides, introduced a method called the **Drivetrain Approach** for thinking about this issue.

    - The **Drivetrain Approach**, illustrated in <>, was described in detail in ["Designing Great Data Products"](https://www.oreilly.com/radar/drivetrain-approach-data-products/). The basic idea is to 
        - Defined Objective: start with considering your clear objective, 
        - Levers: then think about what actions you can take (levers you can pull) to meet that objective and 
        - Data: what data you have (or can acquire) that can help to take that action/pull lever, and then 
        - Models: build a predictive model (of how the levers and other uncontrollable inputs influence the output = predicted state of our objective) 
            - that you can use to determine the best actions to take to get the best results in terms of your objective.

- We use data not just to generate more data (in the form of predictions), but to produce actionable outcomes. That is the goal of the Drivetrain Approach. 

- Searching (DuckDuckGo) for images to process:

    ```python
    bear_types = 'grizzly','black','teddy' # three types
    path = Path('bears')

    if not path.exists():
        path.mkdir()
        for o in bear_types: # put some of each type in directory with that name
            dest = (path/o)
            dest.mkdir(exist_ok=True)
            results = search_images_ddg(f'{o} bear') # ddg search syntax (default max_images = 200)
            download_images(dest, urls=results) # for ddg search
            # download_images(dest, urls=results.attrgot('contentUrl')) # for Bing search
    ```

    ```python
    failed = verify_images(fns)
    failed
    ```

**Data Augmentation:** 
We don't have a lot of data for our problem (150 pictures of each sort of bear at most), so to train our model, we'll use `RandomResizedCrop` with an image size of 224 px, which is fairly standard for image classification, and default aug_transforms:

    ```python
    bears = bears.new(
        item_tfms=RandomResizedCrop(224, min_scale=0.5),
        batch_tfms=aug_transforms())
    dls = bears.dataloaders(path)
    ```

We can now create our Learner and fine-tune it in the usual way:

    ```python
    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(4)
    ```

To visualize performance with a confusion matrix:

    ```python
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    ```

- It's helpful to see where exactly our errors are occurring.
- To do this, we can sort our images by their loss.
    - The loss is a number that is higher if the model is incorrect (especially if it's also confident of its incorrect answer), or if it's correct, but not confident of its correct answer. 

    ```python
    # As the title of the output says, each image is labeled with four things: prediction, actual (target label), loss, and probability. 
    interp.plot_top_losses(5, nrows=1)
    ```

- fastai includes a handy GUI for data cleaning called ImageClassifierCleaner that allows you to choose a category and the training versus validation set and view the highest-loss images (in order), along with menus to allow images to be selected for removal or relabeling:

    ```python
    cleaner = ImageClassifierCleaner(learn)
    cleaner

    # for idx in cleaner.delete(): cleaner.fns[idx].unlink() # delete those marked for del
    # for idx,cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path/cat) # move those relabeled
    ```

#### Saving and restoring a learned model

    ```python
    learn.export('model.pkl') # save a model
    # ...
    learn_inf = load_learner(path/'export.pkl') # load a saved model
    # to get predictions for one image at a time, pass a filename to predict
    learn_inf.predict('images/grizzly.jpg')
    learn_inf.dls.vocab # vocab of the DataLoaders is a stored list of all possible categories
    ```
To do inference with an estimated/fitted model, pass the filename to `predict`:

    ```python
    learn_inf.predict('images/grizzly.jpg')
    ```

#### Observations
- Cleaning the data and getting it ready for your model are two of the biggest challenges for data scientists; they say it takes 90% of their time. The fastai library aims to provide tools that make it as easy as possible.
- Insight: best to do data cleaning after initial training ("before you clean the data, train the model")
- No Need for Big Data: After cleaning the dataset using these steps, we generally are seeing 100% accuracy on this task.
    - the common complaint that you need massive amounts of data to do deep learning can be a very long way from the truth!
    - if you do need more data, can do Data Augmentation (synthetic data), e.g. with `RandomResizedCrop`

## Saving the Model and Using the Model for Inference

- A good current way to save and host an ML model is with `Gradio` framework, hosted on `HuggingFace Spaces`
- Learn about easy ML app development
    - Tanishq Abraham's blog, Nov 16, 2021 
    - [Gradio + HuggingFace Spaces: A Tutorial for ML app development](https://tmabraham.github.io/blog/gradio_hf_spaces_tutorial) 

#### Basic Ideas for ML Deployment: (from Fastai PDL4Coders Lesson 2 video)
- **Fastai** is a python library that makes easy the construction of a variety of ML models (as a higher-level library calling Pytorch libraries)
    - This is easily done with Jupyter notebooks.
    - Note: Fastai is also accessible from Julia, see the FluxML/Fastai.jl library repository
    - Howard and fastai also developed and recomment\d `nbdev`, a library enhancing Jupyter notebooks
        - [`fastai/nbdev`](https://github.com/fastai/nbdev): "Simply write notebooks with lightweight markup and get high-quality documentation, tests, continuous integration, and packaging for free!"
- **Kaggle** and **Google Colab** both offer online, free workspaces (with GPU) for developing and training ML models, e.g. using Fastai library and Jupyter notebooks
- **HuggingFace Spaces** will host an ML (Data science) model and execute it, e.g., executing any pretrained model to do predictions, via an API
    - the python for such a model can be extracted/exported from a Jupyter notebook with fastai's NBDev tools/enhancements
- **Gradio** allows construction of an interface with widgets for inputs and outputs, for the HuggingFace Spaces model, that executes as a web page. (Gradio just creates Javascript?)
    - an alternative is Streamlit
    - Gradio makes the API viewable, and allows it to be called from other languages and enhances/customized
- Knowledge of **Javascript** allows further customization of the webpage created by Gradio.
    - Note: A javascript file, embedded in HTML, will execute on any machine with a modern browser, with no other software installed.
- **Github Pages (github.io)** is one option for hosting the website (the HTML pages) in the cloud with public or private access
    - [`fastai/fastpages` repository](https://github.com/fastai/fastpages) will help set blog pages with support for Jupyter notebooks
        - "fastpages uses GitHub Actions to simplify the process of creating Jekyll blog posts on GitHub Pages from a variety of input formats."
- **Local Development** To setup, develop and execute such models locally (rather than online), one must install and set up a working version of Python and all necessary libraries for fastai and jupyter notebooks
    - see [`fastai/fastsetup` repository](https://github.com/fastai/fastsetup) for one way of doing this 
        - (Q: will this install separately and cleanly from all exising python/anaconda installations?)
        - These instructions seem to be oriented to Ubuntu OS
    - fastai/fastsetup recommends using the `mamba` package management tool, as sort of a faster-but-compatible `conda`

- For a summary of steps demonstrated in Lesson 2, see the diagram at [Lesson 2: Practical Deep Learning for Coders 2022 - time 1:00:50](https://youtu.be/F4tvM4Vb3A0?t=3650)
    - Create a "Space"
    - Try a Basic (Gradio) Interface
        - set up git, conda (mamba)
    - Try Executable ML model in a Notebook
        - Cat/Dog or Pet Breeds example
    - Use an exported Learner `fastai.learner`
    - Use NBDEV (e.g. to export the key Jupyter cells to create a python file, e.g. app.py, )
    - Host the ML model and assoc webpage on Github 
    


[Lesson 3: Neural net foundations](https://course.fast.ai/Lessons/lesson3.html)
---------------------------------------------------------------------------------
- Biggest #1 mistake of beginners is to jump quickly to bigger model architectures
- Howard likes to start with simple model architecture, learn and improve the data
    - "At the start of a new project I pretty much only use Resnet18
    - sopend time trying things
    - consider data augmentation (e.g. for images, try `RandomResizedCrop`)
    - try cleaning the data
    - try different external data that can be brought in
- Generally, trying different (more elaborate) architectures is the _last_ thing I do

[Lesson 4: Natural Language (NLP)](https://course.fast.ai/Lessons/lesson4.html)
--------------------------------------------------------------------------------

[Lesson 5: From-scratch model](https://course.fast.ai/Lessons/lesson5.html)
------------------------------------------------------------------------

[Lesson 6: Random forests](https://course.fast.ai/Lessons/lesson6.html)
----------------------------------------------------------------

[Lesson 7: Collaborative filtering](https://course.fast.ai/Lessons/lesson7.html)
----------------------------------------------------------------

[Lesson 8: Convolutions (CNNs)](https://course.fast.ai/Lessons/lesson8.html)
---------------------------------------------------------------------------

[Lesson 9: Data ethics](https://course.fast.ai/Lessons/lesson9.html)
----------------------------------------------------------------
