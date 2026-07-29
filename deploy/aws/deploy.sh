#!/usr/bin/env bash
set -Eeuo pipefail

required=(AWS_AMI_ID REPOSITORY_URL)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "Missing required environment variable: ${variable}" >&2
    exit 2
  fi
done

command -v aws >/dev/null || { echo "AWS CLI is required." >&2; exit 2; }
stack_name="${AWS_STACK_NAME:-atlas-metadata-studio}"

aws cloudformation deploy \
  --stack-name "${stack_name}" \
  --template-file deploy/aws/ec2-cloudformation.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    AmiId="${AWS_AMI_ID}" \
    InstanceType="${AWS_INSTANCE_TYPE:-g5.xlarge}" \
    RepositoryUrl="${REPOSITORY_URL}" \
    RepositoryBranch="${REPOSITORY_BRANCH:-main}" \
    AllowedCidr="${ALLOWED_CIDR:-127.0.0.1/32}"

aws cloudformation describe-stacks \
  --stack-name "${stack_name}" \
  --query 'Stacks[0].Outputs' \
  --output table
