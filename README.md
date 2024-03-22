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
pip3 install pynq-dpu --no-build-isolation
```

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

