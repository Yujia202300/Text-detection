* python3.6

* pytorch, cython, python-opencv,  easydict,  etc.

CNN+end to end+multi-scale attention mechanism for image text detection


***


* JPEGImages folder :   Training pictures and test pictures

* Animations folder :  Label file in xml format

* ImageSets folder :  Action is temporarily unavailable  Layout is temporarily unavailable

 * Main stores the data of image object recognition，Main contains test.txt , train.txt, val.txt,trainval.txt.

***
#### Data production
* Put training set pictures into JPRGImages

* xml.py:  Used to make an. xml file

* generate_maintxt.py:  Used to generate. txt files under the Main folder
***
#### Build a Python module
```bash
cd $CNN_ROOT/lib
make
```
***
#### test model
```bash
cd $CNN_ROOT
```

***
#### Training model
```bash
cd $CNN_ROOT

```




