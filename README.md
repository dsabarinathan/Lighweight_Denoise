
# Light Weight Residual Dense for Denoise

## Environment

1. Python 3.6.1
2. Anaconda 5.0.1
3. Ubuntu 16.04 or Windows10
4. Keras 2.2.4
5. tensorflow 1.10.0

## How to setup the environment

### Step1 

Unzip the downloaded folder


### Step2

Open the powershell or terminal


### Step3

```
$cd yourpathtoLightWeightModel

$pwd
> ~/LightWeightModel

$pip install --upgrade -r requirements.txt

```
## How to test the version one model on your own imgaes
```
$python test_v1.py --testImagePath=yourmatfilepath --noisy_key=yournoisekeyname 
```
## How to test the version two model on your own imgaes
```
$python test_v2.py --testImagePath=yourmatfilepath --noisy_key=yournoisekeyname 
```
