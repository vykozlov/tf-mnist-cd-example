node {
  def dockerhubuser = 'vykozlov'
  def appName = 'tf-mnist-cd'
  def imageTag = "${dockerhubuser}/${appName}:${env.BRANCH_NAME}.${env.BUILD_NUMBER}"

  stage ('Clone repository') {
      checkout scm
  }

  stage('Build image') {
      //sh("nvidia-docker build -t ${imageTag} .")
  }

  stage('Run tests') {
      //sh("nvidia-docker run ${imageTag} python tools/tf_vers.py")
  }

  stage('Push image to registry') {
      //sh("docker -- push ${imageTag}")
  }

  stage('Deploy Application') {
      switch (env.BRANCH_NAME) {
      // Roll out to production
        case "master":
            // Change deployed image in canary to the one we just built
            sh("sed -i.bak 's#vykozlov/tf-mnist-cd:1.5.0#${imageTag}#' ./k8s/production/*.yaml")
            sh("kubectl --namespace=production apply -f k8s/services/tf-mnist-cd-svc.yaml")
            sh("kubectl --namespace=production apply -f k8s/production/")
            break

        // Roll out a dev environment
        default:
            // Create namespace if it doesn't exist
            sh("kubectl get ns ${env.BRANCH_NAME} || kubectl create ns ${env.BRANCH_NAME}")
            // Don't use public load balancing for development branches
            sh("sed -i.bak 's#vykozlov/tf-mnist-cd:1.5.0#${imageTag}#' ./k8s/dev/*.yaml")
            sh("kubectl --namespace=${env.BRANCH_NAME} apply -f k8s/services/tf-mnist-cd-dev-svc.yaml")
            sh("kubectl --namespace=${env.BRANCH_NAME} apply -f k8s/dev/")
            //echo 'To access your environment run `kubectl proxy`'
            //echo "Then access your service via http://localhost:8001/api/v1/proxy/namespaces/${env.BRANCH_NAME}/services/${feSvcName}:80/"
      }
  }
}
