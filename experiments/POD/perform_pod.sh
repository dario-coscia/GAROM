mkdir -p singularvalues
mkdir -p results

echo "GAUSSIAN" 
echo "##########################"
python3 pod_gaussian.py --singular_values=1

echo "GRAETZ" 
echo "##########################"
python3 pod_graetz.py --singular_values=1

echo "LID" 
echo "##########################"
python3 pod_lid.py --singular_values=1
