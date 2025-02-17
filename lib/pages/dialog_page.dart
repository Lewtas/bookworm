import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

class DialogPage extends StatefulWidget {
  const DialogPage({Key? key}) : super(key: key);

  @override
  _DialogPageState createState() => _DialogPageState();
}

class _DialogPageState extends State<DialogPage> {
  TextEditingController _sendMessageController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      body: Container(
        decoration: const BoxDecoration(
            gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                stops: [0.3, 0.7],
                colors: [Color(0xccE97EA6), Color(0xccB0EADA)])),
        child: Padding(
          padding: const EdgeInsets.only(top: 110.0),
          child: Column(
            children: [
              Expanded(
                child: SingleChildScrollView(
                  child: Column(
                    children: [
                      getMessage(
                          'Lorem ipsum dolor sit amet,consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.',
                          isRightMessage: false),
                      getMessage('Lorem ipsum dolor sit amet',
                          isRightMessage: true),
                      getMessage('Lorem ipsum dolor sit amet',
                          isRightMessage: false),
                      getMessage(
                          'Lorem ipsum dolor sit amet, nonummy euismod tincid',
                          isRightMessage: true),
                      getMessage('Lorem ipsum dolor sit amet, nonummy euismod',
                          isRightMessage: false),
                      getMessage('Lorem ipsum dolor sit amet',
                          isRightMessage: true),
                      getMessage(
                          'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.',
                          isRightMessage: false),
                    ],
                  ),
                ),
              ),
              getBottom(),
            ],
          ),
        ),
      ),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: TextButton(
          onPressed: () {
            Navigator.pop(context);
          },
          child: const Icon(
            Icons.close,
            color: Colors.white,
            size: 40,
          ),
        ),
        title: Row(
          children: <Widget>[
            Container(
              width: 40,
              height: 40,
              decoration: const BoxDecoration(
                  shape: BoxShape.circle,
                  image: DecorationImage(
                      image: NetworkImage(
                          "https://images.unsplash.com/photo-1531427186611-ecfd6d936c79?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=800&q=60"),
                      fit: BoxFit.cover)),
            ),
            const SizedBox(
              width: 15,
            ),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                const Text(
                  "Tyler Nix",
                  style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Colors.white),
                ),
                const SizedBox(
                  height: 3,
                ),
                Text(
                  "Active now",
                  style: TextStyle(
                      color: Colors.black.withOpacity(0.4), fontSize: 14),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget getMessage(String message, {required bool isRightMessage}) {
    return Padding(
      padding: const EdgeInsets.only(right: 20, left: 20, bottom: 10),
      child: Row(
        mainAxisAlignment:
            isRightMessage ? MainAxisAlignment.end : MainAxisAlignment.start,
        children: [
          Container(
            decoration: BoxDecoration(
              color: isRightMessage ? Color(0xFFB57B94): Color(0xFF794E72),
              borderRadius: BorderRadius.circular(10),
            ),
            child: ConstrainedBox(
              constraints: BoxConstraints(
                maxWidth: MediaQuery.of(context).size.width / 1.5,
              ),
              child: Padding(
                padding: const EdgeInsets.all(8.0),
                child: Text(
                  message,
                  style: TextStyle(color: Colors.white),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget getBottom() {
    return Padding(
      padding: const EdgeInsets.all(8.0),
      child: Container(
        // width: double.infinity,
        decoration: BoxDecoration(
          color: const Color(0xFFE3FDF8),
          borderRadius: BorderRadius.circular(100),
        ),
        child: Row(
          // mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: <Widget>[
            const Padding(
              padding: EdgeInsets.symmetric(horizontal: 8.0),
              child: Icon(
                Icons.add,
                size: 35,
                color: Color(0x99e22678),
              ),
            ),
            Expanded(
              child: TextField(
                cursorColor: Colors.black,
                controller: _sendMessageController,
                decoration: const InputDecoration(
                  border: InputBorder.none,
                  hintText: "Write a message...",
                ),
              ),
            ),
            const Padding(
              padding: EdgeInsets.symmetric(horizontal: 8.0),
              child: Icon(
                Icons.keyboard_voice,
                size: 30,
                color: Color(0x99e22678),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
