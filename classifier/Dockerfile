ARG PROJECT
FROM image-registry.openshift-image-registry.svc:5000/$PROJECT/sp-lr-classifier-image:latest
RUN echo "---> copying source"
COPY . /microservice
WORKDIR /microservice



