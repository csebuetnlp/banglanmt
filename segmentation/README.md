
## Setup

``` python setup.py install``` or ``` pip install .```

## Usage

* ### From python scripts
    ```python
    >>> from segmentation import segmenter
    >>> input_text = '''
        কাজী মুহম্মদ ওয়াজেদের একমাত্র পুত্র ছিলেন এ. কে. ফজলুক হক। A. K. Fazlul Huq (Sher-E-Bangla) was born into a middle class Bengali Muslim family in Bakerganj, Barisal, Bangladesh in 1873. 
    '''
    >>> segmenter.segment_text(input_text)
    ['কাজী মুহম্মদ ওয়াজেদের একমাত্র পুত্র ছিলেন এ. কে. ফজলুক হক।',
    'A. K. Fazlul Huq (Sher-E-Bangla) was born into a middle class Bengali Muslim family in Bakerganj, Barisal, Bangladesh in 1873.']
    ```
    *If you don't want a linebreak to be an explicit new line marker, use the following*
    ```python        
    >>> segmenter.segment_text(input_text, mode='multi') 
    ```
    
        
  * ***Note: the above snippets run with most of the default options, for more advanced options, refer to the terminal script.***      
         
* ### From terminal
    ```bash
    segmenter --help
    ```