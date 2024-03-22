# cis589_proj

## Hardware
* AMD Kria KV260
* Logitech C920 webcam

## Install dependencies
Start from Ubuntu 22.04 on KV260

```bash
sudo apt-get install -y \
  python3-boto3
```

## Install Kria PYNQ DPU
```bash
git clone https://github.com/Xilinx/Kria-PYNQ.git
cd Kria-PYNQ
sudo bash ./install.sh -b KV260; # This will take about 25 minutes
```
Then login as root
```bash
sudo su
```
and complete the installation as root
```bash
. /home/ubuntu/Kria-PYNQ/pynq/sdbuild/packages/xrt/xrt_setup.sh
. /etc/profile.d/pynq_venv.sh
pip3 install pynq-dpu --no-build-isolation
```
then copy the xmodel file
TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

## AWS Setup
1. Must have an AWS IAM user with AmazonRekognition* permission policies (add a group with all those policies then add user to group)
2. Set up target environment to communicate with aws
```bash
mkdir -p ~/.aws
touch ~/.aws/credentials
touch ~/.aws/config
```

where `~/.aws/credentials` looks like
```
[default]
aws_access_key_id = your_access_key_id
aws_secret_access_key = your_secret_access_key
```

and `~/.aws/config` looks like
```
[default]
region = your_aws_region
```

see https://docs.aws.amazon.com/pdfs/rekognition/latest/dg/rekognition-dg.pdf#what-is

## Environment setup
```bash
source /home/ubuntu/Kria-PYNQ/pynq/sdbuild/packages/xrt/xrt_setup.sh
source /etc/profile.d/pynq_venv.sh
```

## Run the cloud app
```bash
./cloud.py -h
```

For example, to detect faces using an image on disk
```bash
./cloud.py --print_duration --display --test_image ./irishterrier-696543.jpg --algorithm detect_faces
```

or to run on the live camera
```bash
./cloud.py --display
```

## Run the edge app
1. Login as root
```bash
sudo su
```
2. As root, setup the environment
```bash
. /home/ubuntu/Kria-PYNQ/pynq/sdbuild/packages/xrt/xrt_setup.sh
. /etc/profile.d/pynq_venv.sh
```
3. Run the script
```bash
./edge.py -h
```

For example, to run the algorithm using an image on disk, print information, and display the result
```bash
./edge.py --print_duration --display --test_image ./irishterrier-696543.jpg
```

or to run on the live camera
```bash
./edge.py --display
```