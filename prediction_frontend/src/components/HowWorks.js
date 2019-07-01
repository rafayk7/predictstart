import React, { Component } from 'react';

export default class HowWorks extends Component {

    render() {
        const questionList = [
            { description: 'How to render list in React?', key: '' },
            { description: 'Do you like JS?', key: ''},
            { description: 'Do you know CSS?', key: '' }
        ];

        return (
            <div>
                <h1>
                    How exactly are predictions made?
            </h1>
                <p>
                    <ol>
                        {questionList.map(question => {
                            return (
                                <li key={question.key}>{question.description}</li>
                            );
                        })}
                    </ol>
                </p>
            </div>

        )
    }

}
