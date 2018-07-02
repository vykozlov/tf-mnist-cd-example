node {
  def dockerhubuser = 'vykozlov'
  def appName = 'tf-mnist-cd'
  def mainVer = "1.0"
  def imageTagBase = "${appName}:${env.BRANCH_NAME}-${mainVer}.${env.BUILD_NUMBER}"
  def imageTag = "${dockerhubuser}/${imageTagBase}-gpu"
  def k8sConfigMaster = "/home/jenkins/.kube/config.master"
  def jname = "jupyter-pass"
  def jpassfile = "./jpassword"  
  
  try {
      stage ('Clone repository') {
          checkout scm
      }

      stage('Build test image and run tests') {
          def imageTagTest = "${imageTagBase}-tests"
          def tmpDirTest = "/tmp/${imageTagTest}"
          tmpDirTest = tmpDirTest.replaceAll(":","-")
          sh("mkdir ${tmpDirTest}")
          docker.build("${imageTagTest}", "-f Dockerfile.tests ./")
          docker.image("${imageTagTest}").withRun("-v $tmpDirTest:/tmp ./run_pylint.sh >/tmp/pylint.log") {
              echo "running container"
              cat ${tmpDirTest}/pylint.log
          }
          
          //warnings canComputeNew: false, canResolveRelativePaths: false, categoriesPattern: '', defaultEncoding: '', excludePattern: '', healthy: '', includePattern: '', messagesPattern: '', parserConfigurations: [[parserName: 'PyLint', pattern: '**/pylint.log']], unHealthy: ''
          echo "Here should be more tests for ${imageTagTest}"
      }

      stage('Build and Push (gpu)image to registry') {
          echo "${imageTag}"
          sh("nvidia-docker build -t ${imageTag} .")
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
  } catch (e) {
    // If there was an exception thrown, the build failed
    currentBuild.result = "FAILED"
    throw e
  } finally {
    // Success or failure, always send notifications
    notifyBuild()
  }
}

def notifyBuild() {
    String buildStatus =  currentBuild.result
  
    // One can re-define default values
    def subject = "${buildStatus}: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]'"
    def summary = "${subject} (${env.BUILD_URL})"
    def details = """<p>STARTED: Job '${env.JOB_NAME} - build # ${env.BUILD_NUMBER}' on $env.NODE_NAME.</p>
      <p>TERMINATED with: ${buildStatus}
      <p>Check console output at "<a href="${env.BUILD_URL}">${env.BUILD_URL}</a>"</p>"""


    emailext (
        subject: '${DEFAULT_SUBJECT}', //subject,
        mimeType: 'text/html',
        body: details,                 //'${DEFAULT_CONTENT}'
        recipientProviders: [[$class: 'CulpritsRecipientProvider'],
                            [$class: 'DevelopersRecipientProvider'],
                            [$class: 'RequesterRecipientProvider']]
    )
}
