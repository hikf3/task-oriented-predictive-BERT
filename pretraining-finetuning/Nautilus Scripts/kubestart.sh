#Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
mkdir -p ~/.local/bin
mv ./kubectl ~/.local/bin/kubectl
export PATH="~/.local/bin:$PATH"
kubectl version --client --output=yaml
#check if installed correctly
kubectl cluster-info -n gpn-mizzou-nextgen-bmi

#Move the config  file to ~/.local/bin
mkdir ~/.kube
mv config ~/.kube/

