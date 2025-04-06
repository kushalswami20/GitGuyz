// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract UserDoctorProfile {
    struct User {
        string name;
        uint age;
        string gender;
        bool isRegistered;
        uint256 tokens;
        uint256 lastLogin;
    }

    struct Doctor {
        string name;
        string specialization;
        uint experience;
        bool isRegistered;
    }

    struct Task {
        string description;
        bool isCompleted;
        address assignedUser;
        uint256 reward;
    }

    mapping(address => User) public users;
    mapping(address => Doctor) public doctors;
    mapping(address => Task) public tasks;

    event UserRegistered(address indexed user, string name, uint age, string gender);
    event DoctorRegistered(address indexed doctor, string name, string specialization, uint experience);
    event TaskAssigned(address indexed doctor, address indexed user, string description, uint256 reward);
    event TaskCompleted(address indexed user, string description, uint256 reward);
    event DailyTokenReceived(address indexed user, uint256 amount);

    uint256 public dailyTokenReward = 10; // Tokens for logging in daily

    modifier onlyRegisteredUser() {
        require(users[msg.sender].isRegistered, "Not a registered user");
        _;
    }

    modifier onlyRegisteredDoctor() {
        require(doctors[msg.sender].isRegistered, "Not a registered doctor");
        _;
    }

    function registerUser(string memory _name, uint _age, string memory _gender) public {
        require(!users[msg.sender].isRegistered, "User already registered");
        require(!doctors[msg.sender].isRegistered, "Address already registered as Doctor");

        users[msg.sender] = User(_name, _age, _gender, true, 0, block.timestamp);
        emit UserRegistered(msg.sender, _name, _age, _gender);
    }

    function registerDoctor(string memory _name, string memory _specialization, uint _experience) public {
        require(!doctors[msg.sender].isRegistered, "Doctor already registered");
        require(!users[msg.sender].isRegistered, "Address already registered as User");

        doctors[msg.sender] = Doctor(_name, _specialization, _experience, true);
        emit DoctorRegistered(msg.sender, _name, _specialization, _experience);
    }

    function claimDailyTokens() public onlyRegisteredUser {
        require(block.timestamp >= users[msg.sender].lastLogin + 1 days, "Already claimed today's reward");

        users[msg.sender].tokens += dailyTokenReward;
        users[msg.sender].lastLogin = block.timestamp;

        emit DailyTokenReceived(msg.sender, dailyTokenReward);
    }

    function assignTask(address _user, string memory _description, uint256 _reward) public onlyRegisteredDoctor {
        require(users[_user].isRegistered, "User not registered");

        tasks[_user] = Task(_description, false, _user, _reward);

        emit TaskAssigned(msg.sender, _user, _description, _reward);
    }

    function completeTask() public onlyRegisteredUser {
        require(tasks[msg.sender].assignedUser == msg.sender, "No task assigned to you");
        require(!tasks[msg.sender].isCompleted, "Task already completed");

        users[msg.sender].tokens += tasks[msg.sender].reward;
        tasks[msg.sender].isCompleted = true;

        emit TaskCompleted(msg.sender, tasks[msg.sender].description, tasks[msg.sender].reward);
    }

    function getUserProfile(address _userAddress) public view returns (string memory, uint, string memory, uint256) {
        require(users[_userAddress].isRegistered, "User not registered");
        User memory u = users[_userAddress];
        return (u.name, u.age, u.gender, u.tokens);
    }

    function getDoctorProfile(address _doctorAddress) public view returns (string memory, string memory, uint) {
        require(doctors[_doctorAddress].isRegistered, "Doctor not registered");
        Doctor memory d = doctors[_doctorAddress];
        return (d.name, d.specialization, d.experience);
    }

    function getTask(address _user) public view returns (string memory, bool, uint256) {
        require(users[_user].isRegistered, "User not registered");
        return (tasks[_user].description, tasks[_user].isCompleted, tasks[_user].reward);
    }
}
