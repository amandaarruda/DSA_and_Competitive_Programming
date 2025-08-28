class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        for (int i = 0; i < nums.size(); i++){ // iterates through the first number
            int complement = target - nums[i]; // calculates the complement
            for (int j = i+1; j < nums.size(); j++){ 
                if (nums[j] == complement){ // seeks for the complementright through
                    return {i,j};
                }
            }
        }
        return {};
    }
};
