node {
  def dockerhubuser = 'vykozlov'
  def appName = 'tf-mnist-cd'
  def imageTag = "${dockerhubuser}/${appName}:${env.BRANCH_NAME}.${env.BUILD_NUMBER}-gpu"
  def k8sConfigMaster = "/home/jenkins/.kube/config.master"
  def jname = "jupyter-pass"
  def jpassfile = "./jpassword"  

  stage ('Clone repository') {
      checkout scm
  }

  stage('Build image') {
      sh("nvidia-docker build -t ${imageTag} .")
  }

  stage('Run tests') {
      //sh("nvidia-docker run ${imageTag} python tools/tf_vers.py")
      echo "Here should be some tests.."
  }

  stage('Push image to registry') {
      echo "${imageTag}"
      withCredentials([usernamePassword(credentialsId: 'dockerhub-vykozlov-credentials', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
        sh '''
          docker login -u ${USERNAME} -p ${PASSWORD}
          '''
      }
      sh("docker push ${imageTag}")     
  }

  stage('Deploy Application') {
      switch (env.BRANCH_NAME) {
      // Roll out to production
        case "master":
            // Change deployed image in canary to the one we just built
            withCredentials([usernamePassword(credentialsId: 'jupyter-credentials', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
              //how to pass parameters here??
              sh '''
                echo -n ${PASSWORD} > ./jpassword
                '''
            }
            sh("kubectl --kubeconfig=${k8sConfigMaster} create secret generic ${jname} --from-file=${jpassfile} --namespace=production --dry-run -o json >${jpassfile}.yaml")
            sh("kubectl --kubeconfig=${k8sConfigMaster} apply -f ${jpassfile}.yaml")
            sh("rm ${jpassfile} ${jpassfile}.yaml")
            sh("sed -i.bak 's#vykozlov/tf-mnist-cd:1.5.0-gpu#${imageTag}#' ./k8s/production/*.yaml")
            sh("kubectl --kubeconfig=${k8sConfigMaster} --namespace=production apply -f k8s/services/tf-mnist-cd-svc.yaml")
            sh("kubectl --kubeconfig=${k8sConfigMaster} --namespace=production apply -f k8s/production/")
            break

        // Roll out a dev environment
        default:
            // Create namespace if it doesn't exist
            sh("kubectl --kubeconfig=${k8sConfigMaster} get ns ${env.BRANCH_NAME} || kubectl --kubeconfig=${k8sConfigMaster} create ns ${env.BRANCH_NAME}")
            withCredentials([usernamePassword(credentialsId: 'jupyter-credentials', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
              sh '''
                echo -n ${PASSWORD} > ./jpassword
                '''
            }
            sh("kubectl --kubeconfig=${k8sConfigMaster} create secret generic ${jname} --from-file=${jpassfile} --namespace=${env.BRANCH_NAME} --dry-run -o json >${jpassfile}.yaml")
            sh("kubectl --kubeconfig=${k8sConfigMaster} apply -f ${jpassfile}.yaml")
            sh("rm ${jpassfile} ${jpassfile}.yaml")       
            // Don't use public load balancing for development branches
            sh("sed -i.bak 's#vykozlov/tf-mnist-cd:1.5.0-gpu#${imageTag}#' ./k8s/dev/*.yaml")
            sh("kubectl --kubeconfig=${k8sConfigMaster} --namespace=${env.BRANCH_NAME} apply -f k8s/services/tf-mnist-cd-dev-svc.yaml")
            sh("kubectl --kubeconfig=${k8sConfigMaster} --namespace=${env.BRANCH_NAME} apply -f k8s/dev/")
            //echo 'To access your environment run `kubectl proxy`'
            //echo "Then access your service via http://localhost:8001/api/v1/proxy/namespaces/${env.BRANCH_NAME}/services/${feSvcName}:80/"
      }
  }

  stage('Post Deploymnet') {
      // delete docker image from Jenkins site
      sh("docker rmi ${imageTag}")
  }

}
