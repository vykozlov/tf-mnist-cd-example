# tf-mnist-cd-example
Example on MNIST Tensorflow script + Docker + k8s + Jenkins

Structure:

apps    - directory containing Tensorflow application + misc tools.

jenkins - for now obsolete

jupyter - copy of Tensorflow jupyter_notebook_config.py + run_jupyter.sh

  * jupyter_notebook_config.py :
    * basic jupyter configuration. If PASSWORD env is set, this PASSWORD is used to login to jupyter.
          
  * run_jupyter.sh :
    * according to Tensorflow people, Jupyter has issues with being run directly: https://github.com/ipython/ipython/issues/7062 , this is a little wrapper script.
    * added JupyterCONF environment in order to mount and use 'external' jupyter config, e.g. private key, certificate, user-defined jupyter_config file.
                         
k8s     - configuration files for kubernetes (sets environments, mount paths)

Dockerfile - to build docker image

Dockerfile.test - for testing of the code in Jenkins pipeline

Jenkinsfile - describes multibranch pipeline (very preliminary) 
                         
