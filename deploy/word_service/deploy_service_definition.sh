#!/bin/sh
gcloud endpoints services deploy ../word_service_proto/api_descriptor.pb api_config.yaml
