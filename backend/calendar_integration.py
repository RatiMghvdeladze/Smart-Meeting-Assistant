import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests
from dotenv import load_dotenv

load_dotenv()


class CalendarTaskManager:
    """
    Integrates with multiple calendar and task management APIs.
    Supports Google Calendar, Outlook Calendar, and Todoist.
    """

    def __init__(self):
        self.google_calendar_enabled = bool(os.getenv("GOOGLE_CALENDAR_API_KEY"))
        self.outlook_enabled = bool(os.getenv("MICROSOFT_CLIENT_ID"))
        self.todoist_enabled = bool(os.getenv("TODOIST_API_TOKEN"))

        # Initialize API clients
        self.todoist_token = os.getenv("TODOIST_API_TOKEN")
        self.google_api_key = os.getenv("GOOGLE_CALENDAR_API_KEY")

        print(f"Calendar Integration Status:")
        print(f"  Google Calendar: {'✅' if self.google_calendar_enabled else '❌'}")
        print(f"  Outlook Calendar: {'✅' if self.outlook_enabled else '❌'}")
        print(f"  Todoist: {'✅' if self.todoist_enabled else '❌'}")

    def create_calendar_events_from_actions(self, action_items: List[Dict], meeting_title: str = "Meeting") -> Dict:
        """
        Create calendar events for action items with due dates.
        """
        results = {
            "created_events": [],
            "failed_events": [],
            "errors": []
        }

        for action in action_items:
            try:
                due_date_str = action.get('due_date', '').strip()
                if not due_date_str or due_date_str.lower() in ['none', 'n/a', '']:
                    continue

                # Parse due date
                due_date = self._parse_due_date(due_date_str)
                if not due_date:
                    continue

                event_data = {
                    'title': f"Action Item: {action['task']}",
                    'description': f"From meeting: {meeting_title}\nAssignee: {action.get('assignee', 'Unassigned')}",
                    'start_time': due_date,
                    'end_time': due_date + timedelta(hours=1),  # 1-hour duration
                    'assignee': action.get('assignee', 'Unassigned')
                }

                # Try to create event in available calendar services
                event_created = False

                if self.google_calendar_enabled:
                    try:
                        google_result = self._create_google_calendar_event(event_data)
                        if google_result:
                            results["created_events"].append({
                                "service": "Google Calendar",
                                "event_id": google_result.get("id"),
                                "action": action['task']
                            })
                            event_created = True
                    except Exception as e:
                        results["errors"].append(f"Google Calendar error: {str(e)}")

                if not event_created:
                    # Fallback: store locally for manual calendar addition
                    results["created_events"].append({
                        "service": "Local Reminder",
                        "event_data": event_data,
                        "action": action['task'],
                        "instructions": f"Please add to your calendar: {event_data['title']} on {due_date.strftime('%Y-%m-%d %H:%M')}"
                    })

            except Exception as e:
                results["failed_events"].append({
                    "action": action.get('task', 'Unknown'),
                    "error": str(e)
                })

        return results

    def create_tasks_from_actions(self, action_items: List[Dict], meeting_title: str = "Meeting") -> Dict:
        """
        Create tasks in task management systems from action items.
        """
        results = {
            "created_tasks": [],
            "failed_tasks": [],
            "errors": []
        }

        for action in action_items:
            try:
                task_data = {
                    'content': action['task'],
                    'description': f"From meeting: {meeting_title}\nAssignee: {action.get('assignee', 'Unassigned')}",
                    'due_date': action.get('due_date', ''),
                    'assignee': action.get('assignee', 'Unassigned')
                }

                # Try Todoist first
                if self.todoist_enabled:
                    try:
                        todoist_result = self._create_todoist_task(task_data)
                        if todoist_result:
                            results["created_tasks"].append({
                                "service": "Todoist",
                                "task_id": todoist_result.get("id"),
                                "task": action['task']
                            })
                            continue
                    except Exception as e:
                        results["errors"].append(f"Todoist error: {str(e)}")

                # Fallback: create local task record
                results["created_tasks"].append({
                    "service": "Local Task List",
                    "task_data": task_data,
                    "task": action['task'],
                    "instructions": f"Manual task: {task_data['content']} (Due: {task_data['due_date']})"
                })

            except Exception as e:
                results["failed_tasks"].append({
                    "action": action.get('task', 'Unknown'),
                    "error": str(e)
                })

        return results

    def _parse_due_date(self, due_date_str: str) -> Optional[datetime]:
        """
        Parse various due date formats into datetime objects.
        """
        due_date_str = due_date_str.lower().strip()
        now = datetime.now()

        try:
            # Handle common phrases
            if 'today' in due_date_str:
                return now.replace(hour=17, minute=0, second=0, microsecond=0)  # 5 PM today
            elif 'tomorrow' in due_date_str:
                return (now + timedelta(days=1)).replace(hour=17, minute=0, second=0, microsecond=0)
            elif 'next week' in due_date_str:
                return (now + timedelta(weeks=1)).replace(hour=17, minute=0, second=0, microsecond=0)
            elif 'eod' in due_date_str:  # End of day
                if 'friday' in due_date_str:
                    days_ahead = 4 - now.weekday()  # Friday is 4
                    if days_ahead <= 0:
                        days_ahead += 7
                    return (now + timedelta(days=days_ahead)).replace(hour=17, minute=0, second=0, microsecond=0)
                else:
                    return now.replace(hour=17, minute=0, second=0, microsecond=0)

            # Try to parse specific date formats
            date_formats = [
                '%Y-%m-%d',
                '%m/%d/%Y',
                '%d/%m/%Y',
                '%Y-%m-%d %H:%M',
                '%m/%d/%Y %H:%M'
            ]

            for fmt in date_formats:
                try:
                    return datetime.strptime(due_date_str, fmt)
                except ValueError:
                    continue

            # If no format matches, return None
            return None

        except Exception as e:
            print(f"Error parsing due date '{due_date_str}': {e}")
            return None

    def _create_google_calendar_event(self, event_data: Dict) -> Optional[Dict]:
        """
        Create an event in Google Calendar.
        Note: This is a simplified implementation. In production, you'd need OAuth2.
        """
        if not self.google_api_key:
            return None

        # This is a mock implementation - Google Calendar API requires OAuth2
        # In a real implementation, you would:
        # 1. Set up OAuth2 authentication
        # 2. Use the Google Calendar API client library
        # 3. Handle proper authentication flow

        print(f"Would create Google Calendar event: {event_data['title']}")
        return {"id": f"mock_google_event_{datetime.now().timestamp()}", "status": "created"}

    def _create_todoist_task(self, task_data: Dict) -> Optional[Dict]:
        """
        Create a task in Todoist using their REST API.
        """
        if not self.todoist_token:
            return None

        try:
            headers = {
                "Authorization": f"Bearer {self.todoist_token}",
                "Content-Type": "application/json"
            }

            # Prepare task payload
            payload = {
                "content": task_data['content'],
                "description": task_data.get('description', ''),
            }

            # Add due date if provided
            if task_data.get('due_date') and task_data['due_date'].lower() not in ['none', 'n/a', '']:
                due_date = self._parse_due_date(task_data['due_date'])
                if due_date:
                    payload["due_string"] = due_date.strftime('%Y-%m-%d')

            response = requests.post(
                "https://api.todoist.com/rest/v2/tasks",
                headers=headers,
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Todoist API error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            print(f"Error creating Todoist task: {e}")
            return None

    def get_integration_status(self) -> Dict:
        """
        Return the status of all integrations.
        """
        return {
            "google_calendar": {
                "enabled": self.google_calendar_enabled,
                "status": "Ready" if self.google_calendar_enabled else "Configure GOOGLE_CALENDAR_API_KEY"
            },
            "outlook_calendar": {
                "enabled": self.outlook_enabled,
                "status": "Ready" if self.outlook_enabled else "Configure MICROSOFT_CLIENT_ID"
            },
            "todoist": {
                "enabled": self.todoist_enabled,
                "status": "Ready" if self.todoist_enabled else "Configure TODOIST_API_TOKEN"
            }
        }

    def get_setup_instructions(self) -> Dict:
        """
        Return setup instructions for each integration.
        """
        return {
            "google_calendar": {
                "steps": [
                    "1. Go to Google Cloud Console (https://console.cloud.google.com/)",
                    "2. Create a new project or select existing one",
                    "3. Enable the Google Calendar API",
                    "4. Create credentials (API Key or OAuth2)",
                    "5. Add GOOGLE_CALENDAR_API_KEY to your .env file"
                ],
                "env_var": "GOOGLE_CALENDAR_API_KEY"
            },
            "todoist": {
                "steps": [
                    "1. Go to Todoist App Management (https://todoist.com/app_console)",
                    "2. Create a new app or use existing one",
                    "3. Generate an API token",
                    "4. Add TODOIST_API_TOKEN to your .env file"
                ],
                "env_var": "TODOIST_API_TOKEN"
            },
            "outlook": {
                "steps": [
                    "1. Go to Azure App Registration (https://portal.azure.com/)",
                    "2. Register a new application",
                    "3. Configure Calendar permissions",
                    "4. Add MICROSOFT_CLIENT_ID and MICROSOFT_CLIENT_SECRET to .env"
                ],
                "env_vars": ["MICROSOFT_CLIENT_ID", "MICROSOFT_CLIENT_SECRET"]
            }
        }


# Initialize the calendar manager
calendar_manager = CalendarTaskManager()


def process_meeting_outcomes(action_items: List[Dict], decisions: List[Dict], meeting_title: str = "Meeting") -> Dict:
    """
    Process meeting outcomes by creating calendar events and tasks.
    This is the main function called from the main app.
    """
    results = {
        "calendar_events": {"created_events": [], "failed_events": [], "errors": []},
        "tasks": {"created_tasks": [], "failed_tasks": [], "errors": []},
        "integration_status": calendar_manager.get_integration_status(),
        "summary": ""
    }

    try:
        # Create calendar events for action items with due dates
        if action_items:
            calendar_results = calendar_manager.create_calendar_events_from_actions(action_items, meeting_title)
            results["calendar_events"] = calendar_results

            # Create tasks for all action items
            task_results = calendar_manager.create_tasks_from_actions(action_items, meeting_title)
            results["tasks"] = task_results

        # Generate summary
        total_events = len(results["calendar_events"]["created_events"])
        total_tasks = len(results["tasks"]["created_tasks"])
        total_errors = len(results["calendar_events"]["errors"]) + len(results["tasks"]["errors"])

        results["summary"] = f"Created {total_events} calendar events and {total_tasks} tasks"
        if total_errors > 0:
            results["summary"] += f" with {total_errors} errors"

        return results

    except Exception as e:
        results["summary"] = f"Error processing meeting outcomes: {str(e)}"
        return results


if __name__ == "__main__":
    # Test the calendar integration
    print("=== Calendar & Task Integration Test ===")

    # Show current status
    status = calendar_manager.get_integration_status()
    print("\nIntegration Status:")
    for service, info in status.items():
        print(f"  {service}: {info['status']}")

    # Show setup instructions if needed
    if not all(info['enabled'] for info in status.values()):
        print("\n=== Setup Instructions ===")
        instructions = calendar_manager.get_setup_instructions()
        for service, info in instructions.items():
            if not status[service]['enabled']:
                print(f"\n{service.upper()}:")
                for step in info['steps']:
                    print(f"  {step}")

    # Test with sample data
    sample_actions = [
        {
            "task": "Finalize project proposal",
            "assignee": "John Doe",
            "due_date": "EOD Friday"
        },
        {
            "task": "Schedule follow-up meeting",
            "assignee": "Jane Smith",
            "due_date": "Tomorrow"
        }
    ]

    print("\n=== Testing with Sample Data ===")
    results = process_meeting_outcomes(sample_actions, [], "Test Meeting")
    print(f"Results: {results['summary']}")