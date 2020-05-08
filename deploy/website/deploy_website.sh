#!/bin/sh
gcloud container clusters get-credentials website --zone us-central1
kubectl apply -f website-deployment.yaml
