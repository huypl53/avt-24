pipeline {
  agent { label ' avt-enhancing' }

  stages{
    stage('build'){
      steps{
        sh "chmod +x -R ${env.WORKSPACE}"
        sh './env.sh'
      }
    }

    stage('deploy'){
      steps{
        // Temporarily the executable programs locate at $HOME/bin/
        sh 'mv ./dist/enhancing $HOME/bin/'
      }
    }

    stage('test'){
      steps{
        sh 'echo $PATH'
        sh 'wget https://www.nearmap.com/content/dam/nearmap/aerial-imagery/us/home/boston-downtown-aerial-image.jpg -O /tmp/boston-downtown-aerial-image.jpg'
        sh '$HOME/bin/enhancing /tmp/boston-downtown-aerial-image.jpg /tmp/boston-downtown-aerial-image_result.jpg'
      }
    }
  }
}
