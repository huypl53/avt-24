pipeline {
  agent any

  stages{
    stage('prepare'){
      steps{
        sh ./env.sh
      }
    }

    stage('build'){
      steps{
        sh 'pyinstaller cli.py --onefile -n enhancing'
      }
    }

    stage('deploy'){
      steps{
        sh 'mv dist/enhancing ~/bin/'
      }
    }

    stage('test'){
      steps{
        sh 'wget 'https://www.nearmap.com/content/dam/nearmap/aerial-imagery/us/home/boston-downtown-aerial-image.jpg -O /tmp/boston-downtown-aerial-image.jpg'
        sh 'enhancing /tmp/boston-downtown-aerial-image.jpg /tmp/boston-downtown-aerial-image_result.jpg'
      }
    }
  }
}
